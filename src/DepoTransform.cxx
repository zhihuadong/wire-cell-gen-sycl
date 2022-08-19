/* Comentary:

   This class is one element in a 2 by 2 matrix of classes which start
   very symmetric.

   (Depo, Impact) outer-product (Zipper, Transform)

   The "Depo" class is high level and uses the low level "Impact" class.

   The "Transform" version is newer than the "Zipper" version and
   while it started with a symmetric interface, it must grow out of
   it.

   The "Transform" version is faster than the "Zipper" for most
   realistically complicated event topology.  It will be made faster
   yet but this will break the symmetry:

   1) The "Tranform" version builds a dense array so sending its
   output through the narrow waveform() is silly.

   2) It can be optimized if the array has a channel basis instead of
   a wire one so the "zipper" interface is not fitting.

   3) It may be further optimized by performing its transforms on two
   planes from each face (for two-faced APAs) simultaneously.  This
   requires a different intrerface, possibly by (ab)using
   BinnedDiffusion in a different way than "zipper".

   4) It may be optimized yet further by breaking up the FFTs to do
   smaller ones (in time) on charge (X) FR and then doing the larger
   ones over ER(X)RC(X)RC.  This work also requires extension of PIR.
   This improvement may also be applicable to zipper.

   All of this long winded explantion, which nobody will ever read, is
   "justification" for the otherwise disgusting copy-paste which is
   going on with these two pairs of classes.

 */
#if defined(SYCL_TARGET_HIP) && defined(__CUDA_ARCH__) 
#undef __CUDA_ARCH__
#endif 

#include "WireCellGenSycl/DepoTransform.h"
#include "WireCellGenSycl/ImpactTransform.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/SimpleTrace.h"
#include "WireCellIface/SimpleFrame.h"
#include "WireCellGenSycl/BinnedDiffusion_transform.h"
#include "WireCellUtil/Units.h"
#include "WireCellUtil/Point.h"
#include "omp.h"

#include "WireCellGenSycl/SyclEnv.h"

WIRECELL_FACTORY(GenSyclDepoTransform, WireCell::GenSycl::DepoTransform, WireCell::IDepoFramer, WireCell::IConfigurable)

using namespace WireCell;
using namespace std;

GenSycl::DepoTransform::DepoTransform()
  : m_start_time(0.0 * units::ns)
  , m_readout_time(5.0 * units::ms)
  , m_tick(0.5 * units::us)
  , m_drift_speed(1.0 * units::mm / units::us)
  , m_nsigma(3.0)
  , m_frame_count(0)
  , m_transform("transform_vector")
  , l(Log::logger("sim"))
{
}

GenSycl::DepoTransform::~DepoTransform() {}

void GenSycl::DepoTransform::configure(const WireCell::Configuration& cfg)
{
    auto anode_tn = get<string>(cfg, "anode", "");
    m_anode = Factory::find_tn<IAnodePlane>(anode_tn);

    m_nsigma = get<double>(cfg, "nsigma", m_nsigma);
    bool fluctuate = get<bool>(cfg, "fluctuate", false);
    m_rng = nullptr;
    if (fluctuate) {
        auto rng_tn = get<string>(cfg, "rng", "");
        m_rng = Factory::find_tn<IRandom>(rng_tn);
    }

    std::string dft_tn = get<string>(cfg, "dft", "FftwDFT");
    m_dft = Factory::find_tn<IDFT>(dft_tn); 

    m_readout_time = get<double>(cfg, "readout_time", m_readout_time);
    m_tick = get<double>(cfg, "tick", m_tick);
    m_start_time = get<double>(cfg, "start_time", m_start_time);
    m_drift_speed = get<double>(cfg, "drift_speed", m_drift_speed);
    m_frame_count = get<int>(cfg, "first_frame_number", m_frame_count);

    auto jpirs = cfg["pirs"];
    if (jpirs.isNull() or jpirs.empty()) {
        std::string msg = "must configure with some plane impact response components";
        l->error(msg);
        THROW(ValueError() << errmsg{"Gen::Ductor: " + msg});
    }
    m_pirs.clear();
    for (auto jpir : jpirs) {
        auto tn = jpir.asString();
        auto pir = Factory::find_tn<IPlaneImpactResponse>(tn);
        m_pirs.push_back(pir);
    }

    m_transform = get<string>(cfg, "transform", m_transform);
}
WireCell::Configuration GenSycl::DepoTransform::default_configuration() const
{
    Configuration cfg;

    /// How many Gaussian sigma due to diffusion to keep before truncating.
    put(cfg, "nsigma", m_nsigma);

    /// Whether to fluctuate the final Gaussian deposition.
    put(cfg, "fluctuate", false);

    /// The open a gate.  This is actually a "readin" time measured at
    /// the input ("reference") plane.
    put(cfg, "start_time", m_start_time);

    /// The time span for each readout.  This is actually a "readin"
    /// time span measured at the input ("reference") plane.
    put(cfg, "readout_time", m_readout_time);

    /// The sample period
    put(cfg, "tick", m_tick);

    /// The nominal speed of drifting electrons
    put(cfg, "drift_speed", m_drift_speed);

    /// Allow for a custom starting frame number
    put(cfg, "first_frame_number", m_frame_count);

    /// Name of component providing the anode plane.
    put(cfg, "anode", "");
    /// Name of component providing the anode pseudo random number generator.
    put(cfg, "rng", "");

    /// Plane impact responses
    cfg["pirs"] = Json::arrayValue;

    /// Transform function to use
    put(cfg, "transform", m_transform);

    return cfg;
}

bool GenSycl::DepoTransform::operator()(const input_pointer& in, output_pointer& out)
{
    double td0(0.0), td1(0.0), td2(0.0), td3(.0), t00 ,t01 ;
       double   t000 = omp_get_wtime() ;
    if (!in) {
        out = nullptr;
        return true;
    }
    
    auto depos = in->depos();

       double   t001 = omp_get_wtime() ;
       std::cout<<"p000 : " << t001-t000 <<std::endl ;
    Binning tbins(m_readout_time / m_tick, m_start_time, m_start_time + m_readout_time);
    ITrace::vector traces;
    for (auto face : m_anode->faces()) {
          t00 = omp_get_wtime() ;
        // Select the depos which are in this face's sensitive volume
        IDepo::vector face_depos, dropped_depos;
        auto bb = face->sensitive();
        if (bb.empty()) {
            l->debug("anode {} face {} is marked insensitive, skipping", m_anode->ident(), face->ident());
            continue;
        }

        for (auto depo : (*depos)) {
            if (bb.inside(depo->pos())) {
                face_depos.push_back(depo);
            }
            else {
                dropped_depos.push_back(depo);
            }
        }

        if (face_depos.size()) {
            auto ray = bb.bounds();
            l->debug(
                "anode: {}, face: {}, processing {} depos spanning "
                "t:[{},{}]ms, bb:[{}-->{}]cm",
                m_anode->ident(), face->ident(), face_depos.size(), face_depos.front()->time() / units::ms,
                face_depos.back()->time() / units::ms, ray.first / units::cm, ray.second / units::cm);
        } else continue ;
	
        if (dropped_depos.size()) {
            auto ray = bb.bounds();
            l->debug(
                "anode: {}, face: {}, dropped {} depos spanning "
                "t:[{},{}]ms, outside bb:[{}-->{}]cm",
                m_anode->ident(), face->ident(), dropped_depos.size(), dropped_depos.front()->time() / units::ms,
                dropped_depos.back()->time() / units::ms, ray.first / units::cm, ray.second / units::cm);
        }

          t01 = omp_get_wtime() ;
	  td3+=t01-t00 ;
        int iplane = -1;
        for (auto plane : face->planes()) {
		double t0 = omp_get_wtime() ;
            ++iplane;

            const Pimpos* pimpos = plane->pimpos();

            Binning tbins(m_readout_time / m_tick, m_start_time, m_start_time + m_readout_time);

            GenSycl::BinnedDiffusion_transform bindiff(*pimpos, tbins, m_nsigma, m_rng);
            for (auto depo : face_depos) {
                depo = modify_depo(plane->planeid(), depo);
		if (depo->charge() ==0 ) continue ;
                bindiff.add(depo, depo->extent_long() / m_drift_speed, depo->extent_tran());
		//std::cout<<"Depo: "<< depo->extent_long() <<" "<<depo->extent_tran() << " Drift_speed " <<m_drift_speed <<std::endl ;
            }

            auto& wires = plane->wires();

            auto pir = m_pirs.at(iplane);
            GenSycl::ImpactTransform transform(pir, m_dft, bindiff);

            double t1=omp_get_wtime() ;
	    td0+=t1-t0 ;
	    
	    std::cout<<"debug DepoTransfer: 0 , "<<m_transform<<std::endl;  

	    if(m_transform.compare("transform_vector")==0 ) {
                transform.transform_vector();
            } else if (m_transform.compare("transform_matrix")==0) {
                transform.transform_matrix();
            } else {
                THROW(ValueError() << errmsg{"No transform function named: " + m_transform});
            }

            double t2=omp_get_wtime() ;
	    td1 += t2-t1 ;
            const int nwires = pimpos->region_binning().nbins();
            std::cout<<"nwires: "<<nwires <<" nsamples: "<<tbins.nbins()<< " p1: " << t1-t0 << " p2: "<< t2-t1<< std::endl ;


	    bool cpflag = m_transform.compare("transform_vector")==0 ;

	    auto wfs = transform.waveform_v(nwires, cpflag) ;
	    auto d0 = wfs.extent(0) ;
	    auto wfs_h = wfs.to_host();

//            std::cout<<"Waveform Time: " << timer.seconds() <<std::endl ;
            for (int iwire = 0; iwire < nwires; ++iwire) {
//		std::vector<float>  wave( & wfs_h(0, iwire) , & wfs_h(tbins.nbins(), iwire ) ) ;
		std::vector<float>  wave( & wfs_h[iwire* d0] , & wfs_h[tbins.nbins() +  iwire*d0 ] ) ;

                auto mm = Waveform::edge(wave);
                if (mm.first == (int) wave.size()) {  // all zero
                    continue;
                }

                int chid = wires[iwire]->channel();
                int tbin = mm.first;

                ITrace::ChargeSequence charge(wave.begin() + mm.first, wave.begin() + mm.second);
                auto trace = make_shared<SimpleTrace>(chid, tbin, charge);
                traces.push_back(trace);
            }
             double t3 =omp_get_wtime() ;
	     td2 += t3-t2 ;
            std::cout<<"BD_create_Time: "<< "plane " <<iplane <<" " <<t1-t0<< std::endl;
	    std::cout<<"trasform_maxtrix_Time:  "<< "plane " <<iplane <<" "<<t2-t1<< std::endl ;
            std::cout<< "nwire_loop_Time: "<<"plane " <<iplane<<" " <<t3-t2<<std::endl ;
	    free(wfs_h) ; 
        }
    }
    auto frame = make_shared<SimpleFrame>(m_frame_count, m_start_time, traces, m_tick);
    ++m_frame_count;
    out = frame;
    std::cout<<"Total_DepoTransform::pt0_Time: "<< td3 <<std::endl ;
    std::cout<<"Total_transform_maxtrix_Time: "<< td1 <<std::endl ;
    std::cout<<"Total_BD_create_Time: "<< td0 <<std::endl ;
    std::cout<<"Total_nwire_loop_long_resp_Time: "<< td2 <<std::endl ;
    std::cout<<"Total_DepoTransform::Operator()_Time: "<< omp_get_wtime() - t000 <<std::endl ;
    return true;
}
