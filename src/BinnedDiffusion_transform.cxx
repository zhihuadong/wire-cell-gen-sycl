#include "WireCellGenSycl/BinnedDiffusion_transform.h"
#include "WireCellGenSycl/GaussianDiffusion.h"
#include "WireCellUtil/Units.h"

#include <iostream>             // debug
//#include <omp.h>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <typeinfo>

#include "WireCellGenSycl/SyclEnv.h"
//#include "oneapi/mkl/rng.hpp" 


//This part if need for OMP_RNG
#if defined SYCL_TARGET_CUDA
#define ARCH_CUDA
#elif defined SYCL_TARGET_HIP
#define ARCH_HIP
#else 
#define USE_RANDOM123
#endif



#include "openmp_rng.h" 

//#define MAX_PATCH_SIZE 512 
//#define P_BLOCK_SIZE  512
//#define MAX_PATCHES  50000
//#define MAX_NPSS_DEVICE 1000
//#define MAX_NTSS_DEVICE 1000
//#define FULL_MASK 0xffffffff
//#define RANDOM_BLOCK_SIZE (1024*1024)
//#define RANDOM_BLOCK_NUM 512
//#define MAX_RANDOM_LENGTH (RANDOM_BLOCK_NUM*RANDOM_BLOCK_SIZE)
//#define MAX_RANDOM_LENGTH (MAX_PATCH_SIZE*MAX_PATCHES)
#define PI 3.14159265358979323846

#define MAX_P_SIZE 100 
#define MAX_T_SIZE 100 



using namespace std;
using namespace std::chrono;

using namespace WireCell;

double g_get_charge_vec_time_part1 = 0.0;
double g_get_charge_vec_time_part2 = 0.0;
double g_get_charge_vec_time_part3 = 0.0;
double g_get_charge_vec_time_part4 = 0.0;
double g_get_charge_vec_time_part5 = 0.0;

extern double g_set_sampling_part1;
extern double g_set_sampling_part2;
extern double g_set_sampling_part3;
extern double g_set_sampling_part4;
extern double g_set_sampling_part5;

extern size_t g_total_sample_size;





GenSycl::BinnedDiffusion_transform::BinnedDiffusion_transform(const Pimpos& pimpos, const Binning& tbins,
                                      double nsigma, IRandom::pointer fluctuate,
                                      ImpactDataCalculationStrategy calcstrat)
    : m_pimpos(pimpos)
    , m_tbins(tbins)
    , m_nsigma(nsigma)
    , m_fluctuate(fluctuate)
    , m_calcstrat(calcstrat)
    , m_window(0,0)
    , m_outside_pitch(0)
    , m_outside_time(0)
{
}


GenSycl::BinnedDiffusion_transform::~BinnedDiffusion_transform() {
}





bool GenSycl::BinnedDiffusion_transform::add(IDepo::pointer depo, double sigma_time, double sigma_pitch)
{

    const double center_time = depo->time();
    const double center_pitch = m_pimpos.distance(depo->pos());

    GenSycl::GausDesc time_desc(center_time, sigma_time);
    {
        double nmin_sigma = time_desc.distance(m_tbins.min());
        double nmax_sigma = time_desc.distance(m_tbins.max());

        double eff_nsigma = sigma_time>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in time sigma:"
            //           << " t_depo=" << center_time/units::ms << "ms not in:"
            //           << " t_bounds=[" << m_tbins.min()/units::ms << ","
            //           << m_tbins.max()/units::ms << "]ms"
            //           << " in Nsigma: [" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_time;
            return false;
        }
    }

    auto ibins = m_pimpos.impact_binning();

    GenSycl::GausDesc pitch_desc(center_pitch, sigma_pitch);
    {
        double nmin_sigma = pitch_desc.distance(ibins.min());
        double nmax_sigma = pitch_desc.distance(ibins.max());

        double eff_nsigma = sigma_pitch>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in pitch sigma: "
            //           << " p_depo=" << center_pitch/units::cm << "cm not in:"
            //           << " p_bounds=[" << ibins.min()/units::cm << ","
            //           << ibins.max()/units::cm << "]cm"
            //           << " in Nsigma:[" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_pitch;
            return false;
        }
    }

    // make GD and add to all covered impacts
    // int bin_beg = std::max(ibins.bin(center_pitch - sigma_pitch*m_nsigma), 0);
    // int bin_end = std::min(ibins.bin(center_pitch + sigma_pitch*m_nsigma)+1, ibins.nbins());
    // debug
    //int bin_center = ibins.bin(center_pitch);
    //cerr << "DEBUG center_pitch: "<<center_pitch/units::cm<<endl; 
    //cerr << "DEBUG bin_center: "<<bin_center<<endl;

    auto gd = std::make_shared<GaussianDiffusion>(depo, time_desc, pitch_desc);
    // for (int bin = bin_beg; bin < bin_end; ++bin) {
    //   //   if (bin == bin_beg)  m_diffs.insert(gd);
    //   this->add(gd, bin);
    // }
    m_diffs.push_back(gd);
    return true;
}


void GenSycl::BinnedDiffusion_transform::get_charge_matrix_sycl(SyclArray::array_xxf& out,
                                                                    std::vector<int>& vec_impact, const int start_pitch,
                                                                    const int start_tick)
{

    auto q = GenSycl::SyclEnv::get_queue() ;
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;


    double wstart=0.0, wend = 0.0 , t0 = 0.0 , t1 = 0.0 ;
    ////wstart = omp_get_wtime();
    auto wstart_c = high_resolution_clock::now();

    const auto ib = m_pimpos.impact_binning();

    // set the size of gd  1DArrayand create allocate host memory
    int npatches = m_diffs.size();
    gd_vt gdata(npatches, false);
    auto gdata_h = ( GdData * ) malloc(npatches * sizeof(GdData) ) ;


    // fill the host view data from diffs
    int ii = 0;
    for (auto diff : m_diffs) {
        gdata_h[ii].p_ct = diff->pitch_desc().center;
        gdata_h[ii].t_ct = diff->time_desc().center;
        gdata_h[ii].charge = diff->depo()->charge();
        gdata_h[ii].t_sigma = diff->time_desc().sigma;
        gdata_h[ii].p_sigma = diff->pitch_desc().sigma;
        ii++;
    }

    // copy to device
    q.memcpy(gdata.data(), gdata_h , npatches*sizeof(GdData)).wait() ;

    free(gdata_h) ;

    // make and device friendly Binning  and copy tbin pbin over.
    GenSycl::DBin tb, pb;
    tb.nbins = m_tbins.nbins();
    tb.minval = m_tbins.min();
    tb.binsize = m_tbins.binsize();
    pb.nbins = ib.nbins();
    pb.minval = ib.min();
    pb.binsize = ib.binsize();

    // perform set_sampling_pre tasks in parallel:
    // should we make it into functions
    // create Device 1DArray 
    size_vt np_d(npatches,false);
    size_vt nt_d(npatches,false);
    size_vt offsets_d(npatches * 2,false);
    idx_vt patch_idx(npatches + 1,false);
    db_vt pvecs_d(npatches * MAX_P_SIZE,false);
    db_vt tvecs_d(npatches * MAX_T_SIZE,false);
    db_vt qweights_d(npatches * MAX_P_SIZE,false);
    
    
    //// t0 = omp_get_wtime() ;
    auto t0_c = high_resolution_clock::now();
    std::cout<<"SetSample: View creation:"<<duration_cast<microseconds>(t0_c - wstart_c).count()/1000000.0 <<std::endl ;

    // create host array , because it will be needed on host temperarily.
    // patch_v_h is delayed until we know the size (nt np)


    // Kernel for calculate nt and np and offset_bin s for t and p for each gd
    int nsigma = m_nsigma;
    
    
    //temperary work around , need redesign syclarray without queue, 
    //or they can not be used in kernels
    auto gdata_ptr=gdata.data() ;
    auto nt_d_ptr = nt_d.data() ;
    auto np_d_ptr = np_d.data() ;
    auto offsets_d_ptr = offsets_d.data() ;
    auto patch_idx_ptr = patch_idx.data() ;
    auto pvecs_d_ptr = pvecs_d.data() ;
    auto tvecs_d_ptr = tvecs_d.data() ;
    auto qweights_d_ptr = qweights_d.data() ;
    auto out_ptr = out.data() ;

    q.parallel_for(sycl::range<1>(npatches), [=] ( auto item ) {
	int i =item.get_id(0) ;
        double t_s = gdata_ptr[i].t_ct - gdata_ptr[i].t_sigma * nsigma;
        double t_e = gdata_ptr[i].t_ct + gdata_ptr[i].t_sigma * nsigma;
        int t_ofb = sycl::max(int((t_s - tb.minval) / tb.binsize), 0);
        int ntss = sycl::min((int((t_e - tb.minval) / tb.binsize)) + 1, tb.nbins) - t_ofb;

        double p_s = gdata_ptr[i].p_ct - gdata_ptr[i].p_sigma * nsigma;
        double p_e = gdata_ptr[i].p_ct + gdata_ptr[i].p_sigma * nsigma;
        int p_ofb = sycl::max(int((p_s - pb.minval) / pb.binsize), 0);
        int npss = sycl::min((int((p_e - pb.minval) / pb.binsize)) + 1, pb.nbins) - p_ofb;

        nt_d_ptr[i] = ntss;
        np_d_ptr[i] = npss;
        offsets_d_ptr[i] = t_ofb;
        offsets_d_ptr[npatches + i] = p_ofb;
    }).wait();
   

    //t1 = omp_get_wtime() ;
    auto t1_c = high_resolution_clock::now();
    std::cout<<"SetSample: nt,np :"<<duration_cast<microseconds>(t1_c-t0_c).count()/1000000.0 <<std::endl ;
    std::cout<<"npatches= "<<npatches <<std::endl ;

    //temperary hold nt*np to scan, can we get rid of it ?
    int wg_size = 128;
    int n_wg = (npatches + wg_size -1 )/wg_size ;
    //auto temp  =  sycl::malloc_device<unsigned long > (n_wg*wg_size , q) ;  
    auto temp  =  sycl::malloc_device<unsigned long > (npatches , q) ;  

   /* q.parallel_for(sycl::nd_range<1>(n_wg*wg_size, wg_size), [=] (sycl::nd_item<1> item ) {
 			   int i = item.get_global_linear_id() ;
			   temp[i] = (i< npatches ) ?  np_d_ptr[i] * nt_d_ptr[i] : 0  ;
			   if(  i ==0 ) patch_idx_ptr[0]=0 ;
			   }).wait();
 */
    q.submit( [&] ( sycl::handler &h ) {
    h.parallel_for(sycl::range<1>(npatches), [=] (auto item ) {
			   auto i = item.get_id(0); 
		           temp[i] =  np_d_ptr[i] * nt_d_ptr[i] ;
			   if(  i ==0 ) patch_idx_ptr[0]=0 ;
			   })  ;
    }) ;
    q.wait()  ;
    

    q.parallel_for(sycl::nd_range<1>(n_wg*wg_size, wg_size), [=] (sycl::nd_item<1> item ) {
			   auto wg = item.get_group() ;
			   auto p = joint_inclusive_scan(wg, &temp[0], &temp[npatches],&patch_idx_ptr[1], plus<>() )  ;
                          }).wait();

    unsigned long result ;
    q.memcpy(&result, &patch_idx_ptr[npatches], sizeof(unsigned long) ) ;


    sycl::free(temp, q) ; 
    q.wait();
    ////t0 = omp_get_wtime() ;
    t0_c = high_resolution_clock::now() ;
    std::cout<<"SetSample: scan :"<<duration_cast<microseconds>(t0_c-t1_c).count()/1000000.0 <<std::endl ;
    // debug:
    std::cout << "total patch size: " << result << " WeightStrat: " << m_calcstrat << std::endl;
    // Allocate space for patches on device
    fl_vt patch_d(result, false);
    auto patch_d_ptr = patch_d.data() ;
    // make a 1Darray pointing to random numbers
    unsigned long size = result ;
    wg_size = 32 ;
    if( size % 2 )  size++ ; 
    n_wg = (size/2 + wg_size -1 )/wg_size ;
    size = n_wg*wg_size*2 ;
    SyclArray::Array1D <double > normals(size, false) ; 
    std::cout<<"rand number size: " << size << std::endl ; 
    auto normals_ptr = normals.data() ;
    uint64_t seed = 2020;

    generator_enum gen_type = generator_enum::philox;
    omp_get_rng_normal_double(normals_ptr, size, 0.0f, 1.0f, seed, gen_type);

    ////t1 = omp_get_wtime() ;
    t1_c = high_resolution_clock::now() ;
    std::cout<<"SetSample: rand :"<<duration_cast<microseconds>(t1_c-t0_c).count()/1000000.0 <<std::endl ;
    // decide weight calculation
    int weightstrat = m_calcstrat;

    // each wg resposible for 1 GD , kernel calculate pvecs and tvecs
    //
    //for  GPU
    wg_size = 32 ;

    q.submit([&] (sycl::handler& cgh ) {
      cgh.parallel_for_work_group( sycl::range(npatches), sycl::range(wg_size), [=] (sycl::group<1> g ) {
      	//int ip = g.get_group_id() ;
      	int ip = g.get_id() ;
      	const double sqrt2 = sqrt(2.0);
        int np = np_d_ptr[ip] ;
        int nt = nt_d_ptr[ip];
        double start_t = tb.minval + offsets_d_ptr[ip] * tb.binsize;
        double start_p = pb.minval + offsets_d_ptr[ip + npatches] * pb.binsize;

      	//every thread  for 1 patch.
        // Calculate Pvecs
        if (np == 1) {
	    pvecs_d_ptr[ip * MAX_P_SIZE] = 1.0 ;
        } else {
      	    g.parallel_for_work_item( np, [&](sycl::h_item<1> item ) {
	        auto ii = item.get_local_id()[0] ;
                double step = pb.binsize;
                double start = start_p;
                double factor = sqrt2 * gdata_ptr[ip].p_sigma;
                double x = (start + step * ii - gdata_ptr[ip].p_ct) / factor;
                double ef1 = 0.5 * sycl::erf(x);
                double ef2 = 0.5 * sycl::erf(x + step / factor);
                double val = ef2 - ef1;
                pvecs_d_ptr[ip * MAX_P_SIZE + ii ] = val;
            });
        }

        // Calculate tvecs
        if (nt == 1) {
		tvecs_d_ptr[ip * MAX_T_SIZE] = 1.0;
        }
        else {
            g.parallel_for_work_item( nt, [&](sycl::h_item<1> item ) {
	        auto ii = item.get_local_id()[0] ;
                double step = tb.binsize;
                double start = start_t;
                double factor = sqrt2 * gdata_ptr[ip].t_sigma;
                double x = (start + step * ii - gdata_ptr[ip].t_ct) / factor;
                double ef1 = 0.5 * sycl::erf(x);
                double ef2 = 0.5 * sycl::erf(x + step / factor);
                double val = ef2 - ef1;
                tvecs_d_ptr[ip * MAX_T_SIZE + ii] = val;
            });
        }
        // calculate weights
        if (weightstrat == 2) {
            if (gdata_ptr[ip].p_sigma == 0) { 
	         qweights_d_ptr[ip*MAX_P_SIZE] = (start_p + pb.binsize - gdata_ptr[ip].p_ct) / pb.binsize; 
            }
            else {
                g.parallel_for_work_item(np, [=]( sycl::h_item<1> item ) {
	            auto ii = item.get_local_id()[0] ;
                    double rel1 = (start_p + ii * pb.binsize - gdata_ptr[ip].p_ct) / gdata_ptr[ip].p_sigma;
                    double rel2 = rel1 + pb.binsize / gdata_ptr[ip].p_sigma;
                    double gaus1 = sycl::exp(-0.5 * rel1 * rel1);
                    double gaus2 = sycl::exp(-0.5 * rel2 * rel2);
                    double wt = -1.0 * gdata_ptr[ip].p_sigma / pb.binsize * (gaus2 - gaus1) / sqrt(2.0 * PI) /
                                    pvecs_d_ptr[ip * MAX_P_SIZE + ii ] +
                                (gdata_ptr[ip].p_ct - (start_p +  (ii + 1) * pb.binsize)) / pb.binsize;
                    qweights_d_ptr[ip*MAX_P_SIZE + ii  ] = -wt;
                });
            }
	}
        } ) ;
    });
    q.wait() ;

    ////t0 = omp_get_wtime() ;
    t0_c = high_resolution_clock::now() ;
    std::cout<<"SetSample: Pvec,Tvec,Weight :"<<duration_cast<microseconds>(t0_c-t1_c).count()/1000000.0 <<std::endl ;
    set_sampling_bat( npatches, nt_d,  np_d,  patch_idx, pvecs_d , tvecs_d, patch_d, normals,gdata  ) ;
    // std::cout << "patch_d: " << typeid(patch_d).name() << std::endl;
    // std::cout << "patch_idx: " << typeid(patch_idx).name() << std::endl;
   //// wend = omp_get_wtime();
    auto wend_c = high_resolution_clock::now() ;
    g_get_charge_vec_time_part4 += duration_cast<microseconds>(wend_c-wstart_c).count()/1000000.0 ;
    std::cout<<"SetSample: SetSamplingBatch :"<<duration_cast<microseconds>(wend_c-t0_c).count()/1000000.0 <<std::endl ;
    std::cout <<"set_sampling_Time: "<< duration_cast<microseconds>(wend_c-wstart_c).count()/1000000.0 <<std::endl;
    
    ////wstart = omp_get_wtime();
    wstart_c = high_resolution_clock::now() ;

    // Scatter Add kernel 
    wg_size = 32 ;
    auto out_d0=out.extent(0) ;
    //std::cout<<"out d0 ="<<out_d0 <<std::endl ;

    q.submit ([&] ( sycl::handler & cgh ) {
        cgh.parallel_for_work_group( sycl::range(npatches), sycl::range(wg_size) , [=] (sycl::group<1> g) {
	    //int ipatch = g.get_group_id()[0] ;
	    int ipatch = g.get_id()[0] ;
            int np = np_d_ptr[ipatch];
            int nt = nt_d_ptr[ipatch];
            int p = offsets_d_ptr[npatches + ipatch]-start_pitch;
            int t = offsets_d_ptr[ipatch]-start_tick;
            int patch_size=np*nt ;
	    g.parallel_for_work_item( patch_size , [=] ( sycl::h_item<1> item ) {
                int ii = item.get_local_id(0) ;
		auto idx = patch_idx_ptr[ipatch] + ii ;
		float charge = patch_d_ptr[idx]; 
		double weight = qweights_d_ptr[ii%np+ipatch*MAX_P_SIZE]; 
		int idx_p = p + ii%np ;
		int idx_t = t + ii/np ;
		size_t out_idx = idx_p + idx_t *out_d0  ;
#if defined(SYCL_TARGET_HIP) || defined(SYCL_TARGET_CUDA) 
		auto out_a1 = sycl::atomic_ref< float, 
				sycl::memory_order::relaxed, 
				sycl::memory_scope::device, 
				sycl::access::address_space::global_space> ( out_ptr[out_idx ]);  
		auto out_a2 = sycl::atomic_ref< float, 
				sycl::memory_order::relaxed, 
				sycl::memory_scope::device, 
				sycl::access::address_space::global_space> ( out_ptr[ out_idx + 1  ]);  

#else
		auto out_a1 = sycl::ext::oneapi::atomic_ref< float, 
				sycl::memory_order::relaxed, 
				sycl::memory_scope::device, 
				sycl::access::address_space::global_space> ( out_ptr[out_idx ]);  
		auto out_a2 = sycl::ext::oneapi::atomic_ref< float, 
				sycl::memory_order::relaxed, 
				sycl::memory_scope::device, 
				sycl::access::address_space::global_space> ( out_ptr[ out_idx + 1  ]);  
#endif

		out_a1.fetch_add((float)(charge*weight) ) ; 
	        out_a2.fetch_add((float)(charge*(1. - weight) ) ) ; 

			    } ) ;

	}) ;	
     } );

    q.wait();


    ////wend = omp_get_wtime();
    wend_c = high_resolution_clock::now() ;
    //free the memory
    //why do it in destructor fail ?
    
    np_d.reset() ;
    nt_d.reset() ;
    offsets_d.reset() ;
    patch_idx.reset();
    pvecs_d.reset()  ;
    tvecs_d.reset()  ;
    qweights_d.reset()  ;
    patch_d.reset()  ;
    normals.reset()  ;
    

    ////t0 = omp_get_wtime();
    t0_c = high_resolution_clock::now() ;

     //std::cout << "yuhw: box_of_one: " << SyclArray::dump_2d_view(out,20) << std::endl;
    // std::cout << "yuhw: DEBUG: out: " << SyclArray::dump_2d_view(out,10000) << std::endl;
    g_get_charge_vec_time_part3 += duration_cast<microseconds>(wend_c - wstart_c).count()/1000000.0;
    cout<<"ScatterAdd_Time : "<<duration_cast<microseconds>(wend_c-wstart_c).count()/1000000.0 << endl ; 
    cout<<"reset Time : "<<duration_cast<microseconds>(t0_c-wend_c).count()/1000000.0 << endl ; 
    cout << "get_charge_matrix_sycl(): Total_ScatterAdd_Time : " << g_get_charge_vec_time_part3 << endl;
    cout << "get_charge_matrix_sycl(): Total_set_sampling_Time : " << g_get_charge_vec_time_part4<< endl ;
    cout << "get_charge_matrix_sycl() : m_fluctuate : " << m_fluctuate << endl;

    cout << "set_sampling(): part1 time :: " << g_set_sampling_part1
         << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
}
/*
void GenSycl::BinnedDiffusion_transform::get_charge_matrix(std::vector<Eigen::SparseMatrix<float>* >& vec_spmatrix, std::vector<int>& vec_impact){
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array # 
  std::map<int,int> map_redimp_vec;
  for (size_t i =0; i!= vec_impact.size(); i++){
    map_redimp_vec[vec_impact[i]] = int(i);
  }

  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact # 
  std::map<int, int> map_imp_redimp;

  //std::cout << ib.nbins() << " " << rb.nbins() << std::endl;
  for (int wireind=0;wireind!=rb.nbins();wireind++){
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int,int> imps_range = m_pimpos.wire_impacts(wireind);
    for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no ++){
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
      
      //  std::cout << imp_no << " " << wireind << " " << wire_imp_no << " " << ib.center(imp_no) << " " << rb.center(wireind) << " " <<  ib.center(imp_no) - rb.center(wireind) << std::endl;
      // std::cout << imp_no << " " << map_imp_ch[imp_no] << " " << map_imp_redimp[imp_no] << std::endl;
    }
  }
  
  int min_imp = 0;
  int max_imp = ib.nbins();


   for (auto diff : m_diffs){
    //    std::cout << diff->depo()->time() << std::endl
    //diff->set_sampling(m_tbins, ib, m_nsigma, 0, m_calcstrat);
    diff->set_sampling(m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    //counter ++;
    
    const auto patch = diff->patch();
    const auto qweight = diff->weights();

    const int poffset_bin = diff->poffset_bin();
    const int toffset_bin = diff->toffset_bin();

    const int np = patch.rows();
    const int nt = patch.cols();
    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
      double weight = qweight[pbin];

      for (int tbin = 0; tbin!= nt; tbin++){
	int abs_tbin = tbin + toffset_bin;
	double charge = patch(pbin, tbin);

	// std::cout << map_redimp_vec[map_imp_redimp[abs_pbin] ] << " " << map_redimp_vec[map_imp_redimp[abs_pbin]+1] << " " << abs_tbin << " " << map_imp_ch[abs_pbin] << std::endl;
	
	vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin] ])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge * weight; 
	vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge*(1-weight);
	
	// if (map_tuple_pos.find(std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin))==map_tuple_pos.end()){
	//   map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin)] = vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).size();
	//   vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).push_back(std::make_tuple(map_imp_ch[abs_pbin],abs_tbin,charge*weight));
	// }else{
	//   std::get<2>(vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin] ]).at(map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]],map_imp_ch[abs_pbin],abs_tbin)])) += charge * weight;
	// }
	
	// if (map_tuple_pos.find(std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin))==map_tuple_pos.end()){
	//   map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin)] = vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).size();
	//   vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).push_back(std::make_tuple(map_imp_ch[abs_pbin],abs_tbin,charge*(1-weight)));
	// }else{
	//   std::get<2>(vec_vec_charge.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1]).at(map_tuple_pos[std::make_tuple(map_redimp_vec[map_imp_redimp[abs_pbin]+1],map_imp_ch[abs_pbin],abs_tbin)]) ) += charge*(1-weight);
	// }
	
	
      }
    }

    

    
    diff->clear_sampling();
    // need to figure out wire #, time #, charge, and weight ...
   }

   for (auto it = vec_spmatrix.begin(); it!=vec_spmatrix.end(); it++){
     (*it)->makeCompressed();
   }
   
   
  
}

*/

// a new function to generate the result for the entire frame ... 
void GenSycl::BinnedDiffusion_transform::get_charge_vec(std::vector<std::vector<std::tuple<int,int, double> > >& vec_vec_charge, std::vector<int>& vec_impact){

  double wstart =0.0 , wend=0.0 ;

  //wstart = omp_get_wtime();
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array # 

  std::map<int,int> map_redimp_vec;
  std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
  for (size_t i =0; i!= vec_impact.size(); i++){
    map_redimp_vec[vec_impact[i]] = int(i);
    std::unordered_map<long int, int> map_pair_pos;
    vec_map_pair_pos.push_back(map_pair_pos);
  }
  //wend = omp_get_wtime();
  g_get_charge_vec_time_part1 += wend - wstart;
  cout << "get_charge_vec() : part1 running time : " << g_get_charge_vec_time_part1 << endl;


  
  //wstart = omp_get_wtime();
  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact # 
  std::map<int, int> map_imp_redimp;


  for (int wireind=0;wireind!=rb.nbins();wireind++){
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int,int> imps_range = m_pimpos.wire_impacts(wireind);
    for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no ++){
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
    }
  }

  
  int min_imp = 0;
  int max_imp = ib.nbins();
  int counter = 0;

  //wend = omp_get_wtime();
  g_get_charge_vec_time_part2 += wend - wstart;
  cout << "get_charge_vec() : part2 running time : " << g_get_charge_vec_time_part2 << endl;
  

    // set the size of gd  1DArrayand create allocate host memory
    int npatches = m_diffs.size();
    gd_vt gdata(npatches, false);
    auto gdata_h = ( GdData * ) malloc(npatches * sizeof(GdData) ) ;


    // fill the host view data from diffs
    int ii = 0;
    for (auto diff : m_diffs) {
        gdata_h[ii].p_ct = diff->pitch_desc().center;
        gdata_h[ii].t_ct = diff->time_desc().center;
        gdata_h[ii].charge = diff->depo()->charge();
        gdata_h[ii].t_sigma = diff->time_desc().sigma;
        gdata_h[ii].p_sigma = diff->pitch_desc().sigma;
        //    if(diff->pitch_desc().sigma == 0 || diff->time_desc().sigma == 0  ) std::cout<<"sigma-0 patch: " <<ii <<
         //   std::endl ;
	//std::cout<<"Gdata: " << ii<<" "<<gdata_h(ii).p_ct<<" "<<gdata_h(ii).t_ct<<" "<<gdata_h(ii).charge<<" "<<gdata_h(ii).t_sigma<<" "<<gdata_h(ii).p_sigma <<std::endl ;
        ii++;
    }

    // copy to device
    gdata.copy_from( gdata_h );

    // make and device friendly Binning  and copy tbin pbin over.
    GenSycl::DBin tb, pb;
    tb.nbins = m_tbins.nbins();
    tb.minval = m_tbins.min();
    tb.binsize = m_tbins.binsize();
    pb.nbins = ib.nbins();
    pb.minval = ib.min();
    pb.binsize = ib.binsize();

    // perform set_sampling_pre tasks in parallel:
    // should we make it into functions
    // create Device 1DArray 
    size_vt np_d(npatches,false);
    size_vt nt_d(npatches,false);
    size_vt offsets_d(npatches * 2,false);
    idx_vt patch_idx(npatches + 1,false);
    db_vt pvecs_d(npatches * MAX_P_SIZE,false);
    db_vt tvecs_d(npatches * MAX_T_SIZE,false);
    db_vt qweights_d(npatches * MAX_P_SIZE,false);
    

    // create host array , because it will be needed on host temperarily.
    // patch_v_h is delayed until we know the size (nt np)


    // Kernel for calculate nt and np and offset_bin s for t and p for each gd
    int nsigma = m_nsigma;
    
    auto q = GenSycl::SyclEnv::get_queue() ;
    
    //temperary work around , need redesign syclarray without queue, 
    //or they can not be used in kernels
    auto gdata_ptr=gdata.data() ;
    auto nt_d_ptr = nt_d.data() ;
    auto np_d_ptr = np_d.data() ;
    auto offsets_d_ptr = offsets_d.data() ;
    auto patch_idx_ptr = patch_idx.data() ;
    auto pvecs_d_ptr = pvecs_d.data() ;
    auto tvecs_d_ptr = tvecs_d.data() ;
    auto qweights_d_ptr = qweights_d.data() ;

    q.parallel_for(sycl::range<1>(npatches), [=] (auto item) {
	auto i = item.get_id(0) ;
        double t_s = gdata_ptr[i].t_ct - gdata_ptr[i].t_sigma * nsigma;
        double t_e = gdata_ptr[i].t_ct + gdata_ptr[i].t_sigma * nsigma;
        int t_ofb = sycl::max(int((t_s - tb.minval) / tb.binsize), 0);
        int ntss = sycl::min((int((t_e - tb.minval) / tb.binsize)) + 1, tb.nbins) - t_ofb;

        double p_s = gdata_ptr[i].p_ct - gdata_ptr[i].p_sigma * nsigma;
        double p_e = gdata_ptr[i].p_ct + gdata_ptr[i].p_sigma * nsigma;
        int p_ofb = sycl::max(int((p_s - pb.minval) / pb.binsize), 0);
        int npss = sycl::min((int((p_e - pb.minval) / pb.binsize)) + 1, pb.nbins) - p_ofb;

        nt_d_ptr[i] = ntss;
        np_d_ptr[i] = npss;
        offsets_d_ptr[i] = t_ofb;
        offsets_d_ptr[npatches + i] = p_ofb;
    });

    std::cout<<"npatches= "<<npatches <<std::endl ;
    //  kernel calculate index for patch  Can be mergged to previous kernel ?
    auto result =  sycl::malloc_shared<unsigned long > (1, q) ;  // total patches points
    auto temp  =  sycl::malloc_device<unsigned long > (npatches+1, q) ;  //temperary hold nt*np to scan
    int wg_size = 256 ;
    int n_wg = (npatches + wg_size -1 )/wg_size ;
    q.parallel_for(sycl::nd_range<1>(n_wg*wg_size, wg_size), [=] (sycl::nd_item<1> item ) {
			   auto wg = item.get_group() ;
 			   auto i = item.get_global_linear_id() ;
			   temp[i] = (i< npatches ) ?  np_d_ptr[i] * nt_d_ptr[i] : 0  ;
			   auto p = joint_inclusive_scan(wg, &temp[0], &temp[npatches],&patch_idx_ptr[1], plus<>() )  ;
			   if(i==0 ) result[0]= patch_idx_ptr[npatches] ;

                          });
    sycl::free(temp, q) ; 

    // debug:
    std::cout << "total patch size: " << result[0] << " WeightStrat: " << m_calcstrat << std::endl;
    // Allocate space for patches on device
    fl_vt patch_d(result[0], false);
    // make a 1Darray pointing to random numbers
    unsigned long size = result[0] ;
    if( size % 2 )  size++ ; 
    SyclArray::Array1D <double > normals(size, false) ; 
    auto normals_ptr = normals.data() ;
    // temp space to hold uniform rn
    //auto temp_ur = sycl::malloc_device<double>(size, q) ;
    uint64_t seed = 2020;
//#if defined(SYCL_TARGET_CUDA) || defined(SYCL_TARGET_HIP) 
//#warning: SYCL_TARGET_CUDA || SYCL_TARGET_HIP
    generator_enum gen_type = generator_enum::philox;
//#else
//#warning: host-target
//    generator_enum gen_type = generator_enum::mt19937 ;
//#endif

    omp_get_rng_normal_double(normals_ptr, size, 0.0f, 1.0f, seed, gen_type);
    //oneapi::mkl::rng::philox4x32x10 engine(q, seed); 
    //generate Uniform distribution 
   // oneapi::mkl::rng::uniform<double >  distr(0.0, 1.0);
   // oneapi::mkl::rng::generate(distr, engine, size, temp_ur); 
   // //box muller approx to normal distribution
   // q.parallel_for(size/2 , [=](auto i) { 
   //     normals_ptr[2*i]     = sqrt(-2*sycl::log(temp_ur[i])) * sycl::cos(2*PI*temp_ur[i+1]);
   //     normals_ptr[2*i + 1] = sqrt(-2*sycl::log(temp_ur[i])) * sycl::sin(2*PI*temp_ur[i+1]);	
    // } ) ;
   // sycl::free(temp_ur, q)  ;

    // decide weight calculation
    int weightstrat = m_calcstrat;

    // each wg resposible for 1 GD , kernel calculate pvecs and tvecs
    //
    //for  GPU
    wg_size = 32 ;



    q.submit([&] (sycl::handler& cgh ) {
      cgh.parallel_for_work_group( sycl::range(npatches), sycl::range(wg_size), [=] (sycl::group<1> g ) {
      	//int ip = g.get_group_id() ;
      	int ip = g.get_id() ;
      	const double sqrt2 = sqrt(2.0);
        int np = np_d_ptr[ip] ;
        int nt = nt_d_ptr[ip];
        double start_t = tb.minval + offsets_d_ptr[ip] * tb.binsize;
        double start_p = pb.minval + offsets_d_ptr[ip + npatches] * pb.binsize;

      	//every thread  for 1 patch.
        // Calculate Pvecs
        if (np == 1) {
	    pvecs_d_ptr[ip * MAX_P_SIZE] = 1.0 ;
        } else {
      	    g.parallel_for_work_item( np, [&](sycl::h_item<1> item ) {
	        auto ii = item.get_local_id()[0] ;
                double step = pb.binsize;
                double start = start_p;
                double factor = sqrt2 * gdata_ptr[ip].p_sigma;
                double x = (start + step * ii - gdata_ptr[ip].p_ct) / factor;
                double ef1 = 0.5 * sycl::erf(x);
                double ef2 = 0.5 * sycl::erf(x + step / factor);
                double val = ef2 - ef1;
                pvecs_d_ptr[ip * MAX_P_SIZE + ii ] = val;
                //	   if(ip==0 && ii==0 ) printf("pvecs(0)=%f %f %f %f %f %f\n", val,start_p,gdata(0).p_sigma,
                //factor, ef1, ef2);
            });
        }

        // Calculate tvecs
        if (nt == 1) {
		tvecs_d_ptr[ip * MAX_T_SIZE] = 1.0;
        }
        else {
            g.parallel_for_work_item( nt, [&](sycl::h_item<1> item ) {
	        auto ii = item.get_local_id()[0] ;
                double step = tb.binsize;
                double start = start_t;
                double factor = sqrt2 * gdata_ptr[ip].t_sigma;
                double x = (start + step * ii - gdata_ptr[ip].t_ct) / factor;
                double ef1 = 0.5 * sycl::erf(x);
                double ef2 = 0.5 * sycl::erf(x + step / factor);
                double val = ef2 - ef1;
                tvecs_d_ptr[ip * MAX_T_SIZE + ii] = val;
            });
        }
        // calculate weights
        if (weightstrat == 2) {
            if (gdata_ptr[ip].p_sigma == 0) { 
	         qweights_d_ptr[ip*MAX_P_SIZE] = (start_p + pb.binsize - gdata_ptr[ip].p_ct) / pb.binsize; 
            }
            else {
                g.parallel_for_work_item(np, [&]( sycl::h_item<1> item ) {
	            auto ii = item.get_local_id()[0] ;
                    double rel1 = (start_p + ii * pb.binsize - gdata_ptr[ip].p_ct) / gdata_ptr[ip].p_sigma;
                    double rel2 = rel1 + pb.binsize / gdata_ptr[ip].p_sigma;
                    double gaus1 = sycl::exp(-0.5 * rel1 * rel1);
                    double gaus2 = sycl::exp(-0.5 * rel2 * rel2);
                    double wt = -1.0 * gdata_ptr[ip].p_sigma / pb.binsize * (gaus2 - gaus1) / sqrt(2.0 * PI) /
                                    pvecs_d_ptr[ip * MAX_P_SIZE + ii ] +
                                (gdata_ptr[ip].p_ct - (start_p +  (ii + 1) * pb.binsize)) / pb.binsize;
                    qweights_d_ptr[ip*MAX_P_SIZE + ii  ] = -wt;
		//    if ( ip == 0 ) printf( "KWeight: =%f \n", -wt ) ;
                });
//	    if ( ip == 0 ) printf( "KWeight: =%f, %f \n",  qweights_d_ptr[0], gdata_ptr[ip].p_sigma) ;
            }
	}     
        } ) ;
    });

  //wstart = omp_get_wtime();
    set_sampling_bat( npatches, nt_d,  np_d,  patch_idx, pvecs_d , tvecs_d, patch_d, normals,gdata  ) ;
 
  // std::cout << "patch_d: " << typeid(patch_d).name() << std::endl;
  // std::cout << "patch_idx: " << typeid(patch_idx).name() << std::endl;
  //wend = omp_get_wtime();
  g_get_charge_vec_time_part4 += wend-wstart ;
  std::cout <<"set_sampling_Time: "<< wend-wstart <<std::endl;
  
 
  //copy diffs patch , np,nt,weight,index backup to host
  auto patch_idx_v_h = patch_idx.to_host() ;
  auto np_v_h = np_d.to_host() ;
  auto nt_v_h = nt_d.to_host() ;
  auto qweights_v_h = qweights_d.to_host() ;
  auto patch_v_h = patch_d.to_host() ;
  auto offsets_v_h = offsets_d.to_host() ;
  //wstart = omp_get_wtime();

  cout << "get_charge_vec() : get_charge_vec() set_sampling_bat() time " << wstart-wend<< endl;
  std::cout<<"debug: patch values: "<<m_diffs.size() << " "<< patch_idx_v_h[m_diffs.size()] <<std::endl ;

  g_get_charge_vec_time_part4 += wstart-wend ;
  //for( long int jj=0 ; jj<m_patch_idx_h[m_diffs.size()] ; jj++ )
//  for( long int jj=0 ; jj<100 ; jj++ )
  //for( long int jj=0 ; jj<patch_idx_v_h[m_diffs.size()] ; jj++ )
// for ( int i1=0 ; i1< m_diffs.size() ; i1++) 
//	for (long int jj= patch_idx_v_h[i1] ; jj< patch_idx_v_h[i1+1] ; jj++) 
//  std::cout<<"PatchValues: "<<i1<< " " <<jj-patch_idx_v_h[i1]<<" "<< patch_v_h[jj] << std::endl ;
//  std::cout<<"PatchValues: "<<jj<<" "<< patch_v_h[jj] << std::endl ;

  /*

  int idx=0 ;
  for (auto diff : m_diffs){
 
     
    auto patch = diff->get_patch();
    const auto qweight = diff->weights();

    memcpy(&(patch.data()[0]), &m_patch_h[m_patch_idx_h[idx]], (m_patch_idx_h[idx+1]-m_patch_idx_h[idx]) * sizeof(float));
    idx++ ;

*/

  int idx=0 ;
  for (auto diff : m_diffs){

    //const int poffset_bin = diff->poffset_bin();
    //const int toffset_bin = diff->toffset_bin();
    const int poffset_bin = offsets_v_h[npatches + idx ];
    const int toffset_bin = offsets_v_h[idx] ;

    //const int np = patch.rows();
    //const int nt = patch.cols();
    const int np = np_v_h[idx];
    const int nt = nt_v_h[idx];
     //std::cout << "DEBUG: "
     //<< " imp offset: " << poffset_bin << ", " << toffset_bin
     //<< " ch offset: " << map_imp_ch[poffset_bin] << ", " << toffset_bin
     //<< " np nt: " << np << ", " << nt
     //<< std::endl;

    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
     // double weight = qweight[pbin];
      double weight = qweights_v_h[pbin + idx*MAX_P_SIZE] ;
 //     std::cout<<"DEBUG: weight: "<< weight <<std::endl ;
      // double weight = 1.0 ;
      auto const channel = map_imp_ch[abs_pbin];
      auto const redimp = map_imp_redimp[abs_pbin];
      auto const array_num_redimp = map_redimp_vec[redimp];
      auto const next_array_num_redimp = map_redimp_vec[redimp+1];

      auto& map_pair_pos = vec_map_pair_pos.at(array_num_redimp);
      auto& next_map_pair_pos = vec_map_pair_pos.at(next_array_num_redimp);

      auto& vec_charge = vec_vec_charge.at(array_num_redimp);
      auto& next_vec_charge = vec_vec_charge.at(next_array_num_redimp);

      for (int tbin = 0; tbin!= nt; tbin++){
        int abs_tbin = tbin + toffset_bin;
        //double charge = patch(pbin, tbin);
        double charge = patch_v_h[patch_idx_v_h[idx] + pbin + tbin*np_v_h[idx]];

        long int index1 = channel*100000 + abs_tbin;
        auto it = map_pair_pos.find(index1);
        if (it == map_pair_pos.end()){
          map_pair_pos[index1] = vec_charge.size();
          vec_charge.emplace_back(channel, abs_tbin, charge*weight);
	}else{
          std::get<2>(vec_charge.at(it->second)) += charge * weight;
	}

        auto it1 = next_map_pair_pos.find(index1);
        if (it1 == next_map_pair_pos.end()){
          next_map_pair_pos[index1] = next_vec_charge.size();
          next_vec_charge.emplace_back(channel, abs_tbin, charge*(1-weight));
	}else{
          std::get<2>(next_vec_charge.at(it1->second)) += charge*(1-weight);
	}
	
      }
    }

    if (counter % 5000==0){
      for (auto it = vec_map_pair_pos.begin(); it != vec_map_pair_pos.end(); it++){
	it->clear();
      }
    }

    diff->clear_sampling();
    idx++ ;
  }
  free(patch_idx_v_h) ;
  free(np_v_h) ;
  free(nt_v_h) ; 
  free(qweights_v_h) ;
  free(patch_v_h) ; 
  free(offsets_v_h) ; 
    // std::cout << "yuhw: get_charge_vec dump: \n";
    // for (size_t redimp=0; redimp<vec_vec_charge.size(); ++redimp) {
    //     std::cout << "redimp: " << redimp << std::endl;
    //     auto v = vec_vec_charge[redimp];
    //     for (auto t : v) {
    //        std::cout << get<0>(t) << ", " << get<1>(t) << ", " << get<2>(t) << "\n";
    //     }
    // }
  //wend = omp_get_wtime();
  g_get_charge_vec_time_part3 += wend - wstart;
  cout << "get_charge_vec() :  Total ScaterAdd_Time : " << g_get_charge_vec_time_part3 << endl;
  cout << "get_charge_vec() : set_sampling()_Total_Time : " << g_get_charge_vec_time_part4 << endl;
  cout << "get_charge_vec() : m_fluctuate : " << m_fluctuate << endl;

#ifdef HAVE_CUDA_INC
  cout << "get_charge_vec() CUDA : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3 << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
  cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
#else
  cout << "get_charge_vec() : set_sampling() part1 time : " << g_set_sampling_part1 << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
#endif
}


void GenSycl::BinnedDiffusion_transform::set_sampling_bat(const unsigned long npatches,  
		const size_vt nt_d ,
		const size_vt np_d, 
		const idx_vt patch_idx , 
		const db_vt pvecs_d,
		const db_vt tvecs_d,
	        fl_vt patch_d,
	        const db_vt normals,
	        const gd_vt gdata ) {

  bool fl = false ;
  if( m_fluctuate) fl = true    ;

  auto q = GenSycl::SyclEnv::get_queue() ;
   
  int wg_size = 32 ; 

 
    auto nt_d_ptr = nt_d.data() ;
    auto np_d_ptr = np_d.data() ;
    auto patch_idx_ptr = patch_idx.data() ;
    auto pvecs_d_ptr = pvecs_d.data() ;
    auto tvecs_d_ptr = tvecs_d.data() ;
    auto gdata_ptr = gdata.data() ;
    auto patch_d_ptr = patch_d.data() ;
    auto normals_ptr = normals.data() ;


    q.submit( [&] ( sycl::handler &h ) {
//		    sycl::stream out(npatches*wg_size*2048,2048, h ) ;
  h.parallel_for(sycl::nd_range<1>(npatches*wg_size, wg_size)   , [=]( sycl::nd_item<1> item ) {
      auto wg = item.get_group() ;
      int ip = item.get_group_linear_id()  ;
      int id  = item.get_local_id(0) ;

      int np=np_d_ptr[ip] ;
      int nt=nt_d_ptr[ip] ;
      int patch_size=np*nt ;
      unsigned long p0 =patch_idx_ptr[ip] ;
      
      int n_it = (patch_size + wg_size -1 ) / wg_size ;

      double gsum = 0.0 ; 
      double lsum  = 0.0 ;
      double charge=gdata_ptr[ip].charge ;
      double charge_abs = abs(charge) ;


      for ( int it = 0 ; it < n_it  ; it ++ ) {
          int ii  = it * wg_size + id  ;
	  double v =  ii < patch_size ?  pvecs_d_ptr[ip*MAX_P_SIZE+ ii%np]*tvecs_d_ptr[ip * MAX_T_SIZE + ii/np] : 0.0 ;
	  if( ii<patch_size) {
		  lsum += v; 
		  patch_d_ptr[ii + p0 ] = (float)v ;
    	  }
      }
      //group level sum 
      gsum = reduce_over_group(wg, lsum , plus<>());  
      double r =charge/gsum ;
     
      // normalize to total = charge ;
      for ( int it = 0 ; it < n_it  ; it ++ ) {
          int ii  = it * wg_size + id  ;
	  if (ii < patch_size ) {
		  patch_d_ptr[ii + p0 ] *= (float)r ;
	  	}	  
      }
      if( fl ) {
      	  int n=(int) charge_abs;
          gsum =0.0 ; 
          lsum =0.0 ;
	  for ( int it = 0 ; it < n_it    ; it ++ ) {
	      int ii  = it * wg_size + id  ;
	      if( ii < patch_size ) { 
	       double p =  (double)patch_d_ptr[ii + p0 ]/charge ; //normalized to total 1
	       //if( p <0 || p>1.0 ) out<<"sqrt-: "<<ip<<" "<<ii<<" "<<p0<<" "<<ii+p0<<" \n" ;
	       double q = 1.0-p ;
	       double mu = n*p ;
	       double sigma = sycl::sqrt(p*q*n) ;
	       p = normals_ptr[ii+p0]*sigma + mu ;
	       lsum += p ;
	       patch_d_ptr[ii + p0] = (float) p ;
	      }
	  }
      }
      //normalize again to total charge after fluctuation
     gsum = reduce_over_group(wg, lsum , plus<>()); 
      if( fl ) {
	  for ( int it = 0 ; it < n_it  ; it ++ ) {
              int ii  = it * wg_size + id  ;
	      if (ii < patch_size ) patch_d_ptr[ii+p0] *= float(charge/gsum) ;   
          }
      }

  } ) ;
    } ) ;
    q.wait() ;
}


static
std::pair<double,double> gausdesc_range(const std::vector<GenSycl::GausDesc> gds, double nsigma)
{
    int ncount = -1;
    double vmin=0, vmax=0;
    for (auto gd : gds) {
        ++ncount;

        const double lvmin = gd.center - gd.sigma*nsigma;
        const double lvmax = gd.center + gd.sigma*nsigma;
        if (!ncount) {
            vmin = lvmin;
            vmax = lvmax;
            continue;
        }
        vmin = std::min(vmin, lvmin);
        vmax = std::max(vmax, lvmax);
    }        
    return std::make_pair(vmin,vmax);
}

std::pair<double,double> GenSycl::BinnedDiffusion_transform::pitch_range(double nsigma) const
{
    std::vector<GenSycl::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->pitch_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenSycl::BinnedDiffusion_transform::impact_bin_range(double nsigma) const
{
    const auto ibins = m_pimpos.impact_binning();
    auto mm = pitch_range(nsigma);
    return std::make_pair(std::max(ibins.bin(mm.first), 0),
                          std::min(ibins.bin(mm.second)+1, ibins.nbins()));
}

std::pair<double,double> GenSycl::BinnedDiffusion_transform::time_range(double nsigma) const
{
    std::vector<GenSycl::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->time_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenSycl::BinnedDiffusion_transform::time_bin_range(double nsigma) const
{
    auto mm = time_range(nsigma);
    return std::make_pair(std::max(m_tbins.bin(mm.first),0),
                          std::min(m_tbins.bin(mm.second)+1, m_tbins.nbins()));
}
