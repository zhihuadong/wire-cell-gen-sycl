#include "WireCellGenSycl/ImpactTransform.h"

#include "WireCellAux/DftTools.h"

#include "WireCellUtil/Testing.h"
#include "WireCellUtil/FFTBestLength.h"

#include "WireCellGenSycl/SyclArray.h"

#include <iostream>  // debugging.
//#include <omp.h>
#include <chrono>

using namespace std::chrono ;
using CTime = std::chrono::high_resolution_clock;
template <class Rep, class Period>
constexpr auto F(const std::chrono::duration<Rep,Period>& d)
{
    return std::chrono::duration<double>(d).count();
}

double g_get_charge_vec_time = 0.0;
double g_fft_time = 0.0;

using namespace std;

using namespace WireCell;
GenSycl::ImpactTransform::ImpactTransform(IPlaneImpactResponse::pointer pir,
	       const IDFT::pointer& dft,
       	       BinnedDiffusion_transform& bd)
  : m_pir(pir)
  , m_dft(dft)
  , m_bd(bd)
  , log(Log::logger("sim"))
  {}

bool GenSycl::ImpactTransform::transform_vector()
{
//    double timer_transform = omp_get_wtime();
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    const auto pimpos = m_bd.pimpos();

    //std::cout << "yuhw: transform_vector: \n"
    //<< " m_num_group: " << m_num_group
    //<< " m_num_pad_wire: " << m_num_pad_wire
    //<< " pitch: " << m_pir->pitch()
    //<< " impact: " << m_pir->impact()
    //<< " nwires: " << m_pir->nwires()
    //<< std::endl;
    for (int i = 0; i != m_num_group; i++) {
        double rel_cen_imp_pos;
        if (i != m_num_group - 1) {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i + 1e-9;
        }
        else {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i - 1e-9;
        }
        m_vec_impact.push_back(std::round(rel_cen_imp_pos / m_pir->impact()));
        std::map<int, IImpactResponse::pointer> map_resp;  // already in freq domain

        for (int j = 0; j != m_pir->nwires(); j++) {
            map_resp[j - m_num_pad_wire] = m_pir->closest(rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch());
            Waveform::compseq_t response_spectrum = map_resp[j - m_num_pad_wire]->spectrum();
             //std::cout
             //<< " igroup: " << i
             //<< " rel_cen_imp_pos: " << rel_cen_imp_pos
             //<< " iwire: " << j
             //<< " dist: " << rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch()
             //<< std::endl;
        }
        m_vec_map_resp.push_back(map_resp);

        std::vector<std::tuple<int, int, double> > vec_charge;  // ch, time, charge
        m_vec_vec_charge.push_back(vec_charge);
    }

    // now work on the charge part ...
    // trying to sampling ...
    //double wstart = omp_get_wtime();
    m_bd.get_charge_vec(m_vec_vec_charge, m_vec_impact);
    //double wend = omp_get_wtime();
   // g_get_charge_vec_time += wend - wstart;
    log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total_Time : {}", g_get_charge_vec_time);
    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
    int start_tick = time_range.first - 1;
    int end_tick = time_range.second + 2;
    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

    int npad_wire = 0;
    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);

    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
    m_start_ch = start_ch - npad_wire;
    m_end_ch = end_ch + npad_wire;

    int npad_time = m_pir->closest(0)->waveform_pad();
    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);

    npad_time = ntotal_ticks - end_tick + start_tick;
    m_start_tick = start_tick;
    m_end_tick = end_tick + npad_time;

    //wstart = omp_get_wtime();
    Array::array_xxc acc_data_f_w =
        Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

    int num_double = (m_vec_vec_charge.size() - 1) / 2;

    // speed up version , first five
    for (int i = 0; i != num_double; i++) {
        Array::array_xxc c_data = Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

        // fill normal order
        for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
            c_data(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                   std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                std::get<2>(m_vec_vec_charge.at(i).at(j));
        }
        m_vec_vec_charge.at(i).clear();
        m_vec_vec_charge.at(i).shrink_to_fit();

        // fill reverse order
        int ii = num_double * 2 - i;
        for (size_t j = 0; j != m_vec_vec_charge.at(ii).size(); j++) {
            c_data(end_ch + npad_wire - 1 - std::get<0>(m_vec_vec_charge.at(ii).at(j)),
                   std::get<1>(m_vec_vec_charge.at(ii).at(j)) - m_start_tick) +=
                std::complex<float>(0, std::get<2>(m_vec_vec_charge.at(ii).at(j)));
        }
        m_vec_vec_charge.at(ii).clear();
        m_vec_vec_charge.at(ii).shrink_to_fit();

        // Do FFT on time
        // Do FFT on wire
        c_data = Aux::fwd(m_dft, c_data);

        // std::cout << i << std::endl;
        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();
                // do a inverse FFT
                Waveform::realseq_t rs1_t = Aux::inv_c2r(m_dft,rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Aux::fwd_r2c(m_dft, rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }

            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Aux::inv_c2r(m_dft, rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Aux::fwd_r2c(m_dft, rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Aux::inv_c2r(m_dft, rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Aux::fwd_r2c(m_dft, rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;

            // Do FFT on wire for response // slight larger
            //resp_f_w = Aux::fwd(m_dft, resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            resp_f_w = Aux::fwd(m_dft, resp_f_w, 0);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            c_data = c_data * resp_f_w;
        }

        // Do inverse FFT on wire
        //c_data = Aux::inv(m_dft, c_data, 1);
        c_data = Aux::inv(m_dft, c_data, 0);

        // Add to wire result in frequency
        acc_data_f_w += c_data;
    }


    log->info("yuhw: start-end {}-{} {}-{}",m_start_ch, m_end_ch, m_start_tick, m_end_tick);

    // central region ...
    {
        int i = num_double;
        // fill response array in frequency domain

        Array::array_xxc data_f_w;
        {
            Array::array_xxf data_t_w =
                Array::array_xxf::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            // fill charge array in time-wire domain // slightly larger
            for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
                data_t_w(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                         std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                    std::get<2>(m_vec_vec_charge.at(i).at(j));
            }
            // m_decon_data = data_t_w; // DEBUG tap data_t_w out
            m_vec_vec_charge.at(i).clear();
            m_vec_vec_charge.at(i).shrink_to_fit();

            // Do FFT on time
            data_f_w = Aux::fwd_r2c(m_dft, data_t_w, 1);
            // Do FFT on wire
            data_f_w = Aux::fwd(m_dft, data_f_w, 0);
        }

        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();

                // do a inverse FFT
                Waveform::realseq_t rs1_t = Aux::inv_c2r(m_dft,rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Aux::fwd_r2c(m_dft, rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }
            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Aux::inv_c2r(m_dft, rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Aux::fwd_r2c(m_dft, rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Aux::inv_c2r(m_dft, rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Aux::fwd_r2c(m_dft, rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;
            // Do FFT on wire for response // slight larger
            //resp_f_w = Array::dft_cc(resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            resp_f_w = Aux::fwd(m_dft, resp_f_w, 0);  // Now becomes the f and f in both time and wire domain ...
            data_f_w = data_f_w * resp_f_w;
        }

        // Do inverse FFT on wire
        data_f_w = Aux::inv(m_dft, data_f_w, 0);

        // Add to wire result in frequency
        acc_data_f_w += data_f_w;
    }
    // m_decon_data = Array::idft_cr(acc_data_f_w, 0); // DEBUG only central
    acc_data_f_w = Aux::inv(m_dft, acc_data_f_w, 1);  //.block(npad_wire,0,nwires,nsamples);
    Array::array_xxf real_m_decon_data = acc_data_f_w.real();
    Array::array_xxf img_m_decon_data = acc_data_f_w.imag().colwise().reverse();
    m_decon_data = real_m_decon_data + img_m_decon_data;

//    double timer_fft = omp_get_wtime() - wstart;
//    log->debug("ImpactTransform::transform_vector: FFT_Time: {}", timer_fft);
//    timer_transform = omp_get_wtime() - timer_transform;
//    log->debug("ImpactTransform::transform_vector: Total: {}", timer_transform);
//    g_fft_time += timer_fft ;
//    log->debug("ImpactTransform::transform_vector: Total_FFT_Time: {}", g_fft_time);


    
    log->debug("ImpactTransform::transform_vector: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
    log->debug("transform_vector: m_decon_data.sum(): {}", m_decon_data.sum());

    return true;
}


bool GenSycl::ImpactTransform::transform_matrix()
{
    //double timer_transform = omp_get_wtime();
    auto transform_0 = CTime::now();
    double td0(0.0), td1(.0)   ;
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
    int start_pitch = impact_range.first - 1;
    int end_pitch = impact_range.second + 2;
    if ((end_pitch - start_pitch) % 2 == 1) end_pitch += 1;
    int start_tick = time_range.first - 1;
    int end_tick = time_range.second + 2;
    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

    int npad_wire = 0;
    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);
    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
    m_start_ch = start_ch - npad_wire;
    m_end_ch = end_ch + npad_wire;

    int npad_time = m_pir->closest(0)->waveform_pad();
    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);
    npad_time = ntotal_ticks - end_tick + start_tick;
    m_start_tick = start_tick;
    m_end_tick = end_tick + npad_time;

    int npad_pitch = 0;
    const size_t ntotal_pitches = fft_best_length((end_ch - start_ch + 2 * npad_wire)*(m_num_group - 1), 1);
    npad_pitch = (ntotal_pitches - end_pitch + start_pitch) / 2;
    start_pitch = start_pitch - npad_pitch;
    end_pitch = end_pitch + npad_pitch;
    
     std::cout << "yuhw: "
     << " channel: { " << start_ch << ", " << end_ch << " } "
     << " pitch: { " << start_pitch << ", " << end_pitch << " }"
     << " tick: { " << m_start_tick << ", " << m_end_tick << " }\n";

    // now work on the charge part ...
    // trying to sampling ...
    //double wstart = omp_get_wtime();
    auto wstart = CTime::now();
    td0 += F(wstart-transform_0) ;
    SyclArray::array_xxf f_data = SyclArray::Zero<SyclArray::array_xxf>(end_pitch - start_pitch, m_end_tick - m_start_tick);;
    m_bd.get_charge_matrix_sycl(f_data, m_vec_impact, start_pitch, m_start_tick);
   // double wend = omp_get_wtime();
    auto wend = CTime::now();
    td1 += F( wend-wstart )  ;
    g_get_charge_vec_time += F( wend - wstart );
    log->debug("ImpactTransform::ImpactTransform() : get_charge_matrix() Total_Time :  {}", g_get_charge_vec_time);

    wstart = CTime::now();
    SyclArray::array_xxc acc_data_f_w(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
    SyclArray::array_xxf acc_data_t_w = SyclArray::Zero<SyclArray::array_xxf>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
    log->info("yuhw: pitch   {} {} {} {}",start_pitch, end_pitch, f_data.extent(0), f_data.extent(1));
    log->info("yuhw: start-end {}-{} {}-{} {} {}",m_start_ch, m_end_ch, m_start_tick, m_end_tick, acc_data_f_w.extent(0), acc_data_f_w.extent(1));
    log->info("yuhw: m_num_group {}",m_num_group);

    auto q = GenSycl::SyclEnv::get_queue() ;
    //  Convolution with Field Response
    {
        // fine grained resp. matrix
        SyclArray::array_xxc resp_f_w_k(end_pitch - start_pitch, m_end_tick - m_start_tick);
        int nimpact = m_pir->nwires() * (m_num_group - 1);
        assert(end_pitch - start_pitch >= nimpact);
        double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
        // TODO this loop could be parallerized by seperating it into 2 parts
        // 1, extract the m_pir into a SyclArray first
        // 2, do various operation in a parallerized fasion
	
        SyclArray::Array1D<int> idx_d(nimpact, false) ;
	int * idx_h = (int * ) malloc(sizeof(int)*nimpact) ;
        
	auto ip0 = m_pir->closest(-max_impact + 0.01 * m_pir->impact()) ;
        int sp_size = ip0->spectrum() .size() ;
	SyclArray::array_xxc sp_fs(sp_size,nimpact,false) ;

	std::complex<float> * sps_h = ( std::complex <float>* ) malloc( sp_size * nimpact *sizeof(std::complex<float>) ) ;
	int fillsize = min(sp_size, m_end_tick - m_start_tick );



        for (int jimp = 0; jimp < nimpact; ++jimp) {
            float impact = -max_impact + jimp * m_pir->impact();
             //std::cout << "yuhw: jimp " << jimp << ", " << impact;
            // to avoid exess of boundary.
            // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
            impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();
            auto ip = m_pir->closest(impact);
            Waveform::compseq_t sp_f = ip->spectrum();

	    memcpy((void*) &sps_h[jimp * sp_size] , (void*) &sp_f[0] , sp_size*sizeof(std::complex<float>) ) ; 
           
            // 0 and right on the left; left on the right
            // int idx = impact < 0 ? end_pitch - start_pitch - (std::round(nimpact / 2.) - jimp)
            //                      : jimp - std::round(nimpact / 2.);
            // 0 and left and the left; right on the right
            int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
                                 : end_pitch - start_pitch - (jimp - std::round(nimpact / 2.));
	    idx_h[jimp] = idx ;
        }

        idx_d.copy_from(idx_h);
        sp_fs.copy_from(sps_h);

	std::cout<<"OK0"<<std::endl ;
	auto  sp_ts = SyclArray::idft_cr(sp_fs,1) ;
	std::cout<<"OK"<<std::endl ;
	if (fillsize < sp_size )  
		sp_ts.resize(fillsize, nimpact , 0.0) ;
        auto resp_redu = SyclArray::dft_rc(sp_ts,1) ;

	auto resp_redu_ptr = resp_redu.data(); ;
	        sp_ts.free() ;
	        sp_fs.free() ;

	auto idx_d_ptr =idx_d.data() ;
	auto resp_f_w_k_ptr =resp_f_w_k.data() ;

	q.parallel_for( sycl::range<2>(resp_redu.extent(0), resp_redu.extent(1)), [=] (auto item ) {
			auto i0 = item.get_id(0) ;
			auto i1 = item.get_id(1) ;
			auto d0 = item.get_range(0) ; 
                resp_f_w_k_ptr[idx_d_ptr[i1] + i0*(end_pitch - start_pitch)] = resp_redu_ptr[i0 + i1 * d0  ] ;
            }).wait();


        auto data_d = SyclArray::dft_rc(f_data, 0);
	auto data_d_ptr = data_d.data() ;
	f_data.free()  ;
	q.wait();
	SyclArray::array_xxc data_c(data_d.extent(0), data_d.extent(1) ) ;


        SyclArray::dft_cc(data_d, data_c,  1);
	auto data_c_ptr = data_c.data() ;

	q.wait();
        SyclArray::dft_cc(resp_f_w_k, data_d, 1);
	resp_f_w_k.free() ;


	// S(f) * R(f)
	q.parallel_for( sycl::range<1>(data_c.extent(0)*data_c.extent(1)) , [=] (auto item ) {
		auto i0 = item.get_id(0) ;
	        float a = data_c_ptr[i0].x * data_d_ptr[i0].x - data_c_ptr[i0].y * data_d_ptr[i0].y ;
	        float b = data_c_ptr[i0].x * data_d_ptr[i0].y + data_c_ptr[i0].y * data_d_ptr[i0].x ;
	        data_c_ptr[i0].x = a ; 
	        data_c_ptr[i0].y = b ; 
            }).wait();

	SyclArray::idft_cc(data_c,data_d, 1);
	data_c.free()  ;

//	std::cout<<"Size: data_d: " << data_d.extent(0)<<","<< data_d.extent(1)<<" ; acc_data_f_w: "<< acc_data_f_w.extent(0) <<";" << acc_data_f_w.extent(1) <<std::endl ;
//	std::cout<<"Size: data_c: " << data_c.extent(0)<<","<< data_c.extent(1)<<" ; acc_data_t_w: "<< acc_data_t_w.extent(0) <<";" << acc_data_t_w.extent(1) <<std::endl ;
        // extract M(channel) from M(impact)


	auto acc_data_f_w_ptr = acc_data_f_w.data() ;
	auto d0_acc = data_d.extent(0) ;
	q.parallel_for( sycl::range<2>(acc_data_f_w.extent(0), acc_data_f_w.extent(1)) , [=] (auto item ) {
		auto i0 = item.get_id(0) ;
		auto i1 = item.get_id(1) ;
		auto d0 = item.get_range(0) ; 
               // acc_data_f_w_ptr[i0 + i1 * d0 ] = data_d_ptr[(i0 + 1) * 10 + i1 * d0_acc ];
                acc_data_f_w_ptr[i0 + i1 * d0 ] = data_d_ptr[(i0 ) * 10 + i1 * d0_acc ];
            }).wait();
	free(idx_h) ;
	free(sps_h) ;
	data_d.free() ;
	resp_redu.free() ;
    }



      SyclArray::array_xxf decon_data_v(acc_data_f_w.extent(0),acc_data_f_w.extent(1), false) ;    
      SyclArray::idft_cr(acc_data_f_w, acc_data_t_w, 0); 

    m_decon_data_v = acc_data_t_w ;

     auto acc_data_t_w_h = m_decon_data_v.to_host();
//    std::cout << "yuhw: acc_data_t_w_h: " << SyclArray::dump_2d_view(acc_data_t_w,10) << std::endl;
    //std::cout << "mdeconn_data_dv: (75,1) " << acc_data_t_w_h(75,1) <<std::endl ;
    Eigen::Map<Eigen::ArrayXXf> acc_data_t_w_eigen((float*) acc_data_t_w_h, acc_data_t_w.extent(0), acc_data_t_w.extent(1));
    m_decon_data = acc_data_t_w_eigen; // FIXME: reduce this copy
    std::cout << "mdeconn_data: row, colum " << m_decon_data.rows()<<","<<m_decon_data.cols() <<std::endl ;
    
    //double timer_fft = omp_get_wtime() - wstart;
    double timer_fft = F ( CTime::now() - wstart );
    g_fft_time += timer_fft ;
    std::cout<<"m_decon_data.sum: "<<m_decon_data.rows()<<"," <<m_decon_data.cols()<<",  "<< m_decon_data.sum() <<std::endl ;
    //log->debug("ImpactTransform::transform_matrix: FFT: {}", timer_fft);
    double timer_transform = F(CTime::now() - transform_0);
    log->debug("ImpactTransform::transform_matrix: Total: {}", timer_transform);
    log->debug("ImpactTransform::transform_matrix: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
    log->debug("ImpactTransform::transform_matrix: m_decon_data.sum(): {}", m_decon_data.sum());
    std::cout<<"Tranform_matrix_p0_Time: "<<td0 <<std::endl;
    std::cout<<"get_charge_matrix_Time: "<<td1 <<std::endl;
    std::cout<<"FFTs_Time: "<<timer_fft <<std::endl;
    log->debug("ImpactTransform::transform_matrix: Total_FFT_Time: {}", g_fft_time);
    return true;
}

GenSycl::ImpactTransform::~ImpactTransform() {}

Waveform::realseq_t GenSycl::ImpactTransform::waveform(int iwire) const
{
    const int nsamples = m_bd.tbins().nbins();
    if (iwire < m_start_ch || iwire >= m_end_ch) {
        return Waveform::realseq_t(nsamples, 0.0);
    }
    else {
        Waveform::realseq_t wf(nsamples, 0.0);
        for (int i = 0; i != nsamples; i++) {
            if (i >= m_start_tick && i < m_end_tick) {
                wf.at(i) = m_decon_data(iwire - m_start_ch, i - m_start_tick);
            }
            else {
                // wf.at(i) = 1e-25;
            }
            // std::cout << m_decon_data(iwire-m_start_ch,i-m_start_tick) << std::endl;
        }

        if (m_pir->closest(0)->long_aux_waveform().size() > 0) {
            // now convolute with the long-range response ...
            const size_t nlength = fft_best_length(nsamples + m_pir->closest(0)->long_aux_waveform_pad());

            //nlength = nsamples;

            //std::cout << nlength << " " << nsamples + m_pir->closest(0)->long_aux_waveform_pad() << std::endl;

            wf.resize(nlength, 0);
            Waveform::realseq_t long_resp = m_pir->closest(0)->long_aux_waveform();
            long_resp.resize(nlength, 0);
            Waveform::compseq_t spec = Aux::fwd_r2c(m_dft, wf);
            Waveform::compseq_t long_spec = Aux::fwd_r2c(m_dft, long_resp);
           for (size_t i = 0; i != nlength; i++) {
                spec.at(i) *= long_spec.at(i);
            }
            wf = Aux::inv_c2r(m_dft, spec);
            wf.resize(nsamples, 0);
         }

        return wf;
    }
}

SyclArray::array_xxf GenSycl::ImpactTransform::waveform_v(int nwires, bool cpflag)  
{
	auto q = SyclEnv::get_queue()  ;

    	const int nsamples = m_bd.tbins().nbins();


	size_t nlength = nsamples  ;
    	if (m_pir->closest(0)->long_aux_waveform().size() > 0) {
	   nlength = fft_best_length(nsamples + m_pir->closest(0)->long_aux_waveform_pad()) ;
	}

	SyclArray::array_xxf wfs(nlength, nwires) ;
        auto wfs_ptr =wfs.data();
     
     	   int chs_start = m_start_ch > 0 ? m_start_ch : 0 ;
    	   int chs_end = m_end_ch > nwires ? nwires : m_end_ch ;
     	   int samples_start = m_start_tick >0 ? m_start_tick : 0 ;
     	   int samples_end = m_end_tick > nsamples ? nsamples : m_end_tick ;
	   if(cpflag)  {
		   m_decon_data_v.alloc(m_decon_data.rows(), m_decon_data.cols()) ;
		  m_decon_data_v.copy_from ((float*)m_decon_data.data() ) ;
	   }

	   auto data_ptr = m_decon_data_v.data() ;
	   auto data_d0 = m_decon_data_v.extent(0) ;
	   int ofs_ch = m_start_ch ;
	   int ofs_tick = m_start_tick  ;

	   auto wfs_d0 =  wfs.extent(0) ;
	   auto wfs_d1 =  wfs.extent(1) ;


	   q.parallel_for( sycl::range<2>( (size_t )(samples_end -samples_start), (size_t)(chs_end - chs_start) ) , [=] (auto item ) {
		size_t  i0 = item.get_id(0) + samples_start ;
		size_t  i1 = item.get_id(1) + chs_start  ;
                wfs_ptr[i0 + i1 * wfs_d0  ] = data_ptr [i1 - ofs_ch +  (i0 - ofs_tick ) * data_d0 ] ;
	    } ).wait() ; 


    	if (m_pir->closest(0)->long_aux_waveform().size() > 0) {

	   SyclArray::array_xxc specs(wfs.extent(0), wfs.extent(1) ) ;
     	   SyclArray::dft_rc(wfs, specs , 1) ;
     	   auto specs_ptr = specs.data()  ;

           Waveform::realseq_t long_resp = m_pir->closest(0)->long_aux_waveform();
           long_resp.resize(nlength, 0);
           Waveform::compseq_t long_spec = Aux::fwd_r2c(m_dft, long_resp);
	    
     
	   SyclArray::array_xc  long_spec_d(nlength, false) ;
           auto long_spec_h = (complex <float> * ) malloc( sizeof(complex<float>) * nlength ) ;
     	   memcpy((void*) &long_spec_h[0] , (void*) &long_spec[0] , nlength*sizeof(std::complex<float>) ) ;
	   long_spec_d.copy_from((void * )long_spec_h) ;
	   auto long_spec_d_ptr = long_spec_d.data()  ;

           // S(f) * LongR(f)
	   q.parallel_for(  sycl::range<2>( specs.extent(0), specs.extent(1)) , [=] (auto item ) {
		auto i0 = item.get_id(0)  ;
		auto i1 = item.get_id(1)  ;
		auto d0 = item.get_range(0) ;
                float a =specs_ptr[i0 + i1 *d0 ].x * long_spec_d_ptr[i0].x - specs_ptr[i0 + i1 *d0 ].y * long_spec_d_ptr[i0].y ;
		float b =specs_ptr[i0 + i1 *d0 ].x * long_spec_d_ptr[i0].y + specs_ptr[i0 + i1 *d0 ].y * long_spec_d_ptr[i0].x ;
		specs_ptr[i0 + i1 *d0 ] = {a, b} ;
	    } ).wait() ; 
	   SyclArray::idft_cr(specs,wfs, 1 ); 
      
	   wfs.resize(nsamples, nwires , 0 ) ;


	   free(long_spec_h) ; 
    	}
        return wfs;
    
}
