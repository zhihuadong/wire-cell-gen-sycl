#ifndef WIRECELL_GENSYCL_IMPACTTRANSFORM
#define WIRECELL_GENSYCL_IMPACTTRANSFORM

#ifdef __CUDA_ARCH__
#define __CUDA_PASS__
#undef __CUDA_ARCH__
#endif

#include "WireCellIface/IPlaneImpactResponse.h"
#include "WireCellUtil/Array.h"
#include "WireCellUtil/Logging.h"
#include <Eigen/Sparse>

#ifdef __CUDA_PASS__
#undef __cuda_PASS__
#define __CUDA_ARCH__
#endif



#include "WireCellGenSycl/BinnedDiffusion_transform.h"

namespace WireCell {
    namespace GenSycl {

        /** An ImpactTransform transforms charge on impact positions
         * into waveforms via 2D FFT.
         */
        class ImpactTransform {
            IPlaneImpactResponse::pointer m_pir;
            BinnedDiffusion_transform& m_bd;

            int m_num_group;     // how many 2D convolution is needed
            int m_num_pad_wire;  // how many wires are needed to pad on each side
            std::vector<std::map<int, IImpactResponse::pointer> > m_vec_map_resp;
            std::vector<std::vector<std::tuple<int, int, double> > > m_vec_vec_charge;  // ch, time, charge
            // std::vector<Eigen::SparseMatrix<float>* > m_vec_spmatrix;

            std::vector<int> m_vec_impact;
            Array::array_xxf m_decon_data;
            SyclArray::array_xxf m_decon_data_v;
            int m_start_ch;
            int m_end_ch;
            int m_start_tick;
            int m_end_tick;

            Log::logptr_t log;

           public:
            ImpactTransform(IPlaneImpactResponse::pointer pir, BinnedDiffusion_transform& bd);
            virtual ~ImpactTransform();

            bool transform_vector();

            bool transform_matrix();

            /// Return the wire's waveform.  If the response functions
            /// are just field response (ie, instantaneous current)
            /// then the waveforms are expressed as current integrated
            /// over each sample bin and thus in units of charge.  If
            /// the response functions include electronics response
            /// then the waveforms are in units of voltage
            /// representing the sampling of the output of the FEE
            /// amplifiers.

            // fixme: this should be a forward iterator so that it may cal bd.erase() safely to conserve memory
            Waveform::realseq_t waveform(int wire) const;
	    SyclArray::array_xxf  waveform_v(int wire, bool cpflag ) ;
        };

    }  // namespace GenSycl
}  // namespace WireCell
#endif /* WIRECELL_GENSYCL_IMPACTTRANSFORM */
