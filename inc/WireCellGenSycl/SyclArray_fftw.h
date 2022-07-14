/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_SYCLARRAY_FFTW
#define WIRECELL_KOKKOSARRAY_FFTW

#include <WireCellUtil/Array.h> // tmp solution
#include <iostream> //debug

namespace WireCell {

    namespace KokkosArray {
        /**
         * using eigen based WireCell::Array for fast prototyping
         * this may introduce extra data copying, will investigate later
         * by default eigen is left layout (column major)
         * https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
         * FIXME: float should be Scalar
         */
        thread_local static Eigen::FFT<float> gEigenFFT;
        inline array_xc dft(const array_xf& in)
        {
            Eigen::Map<Eigen::VectorXf> in_eigen((float*) in.data(), in.extent(0));
            Eigen::VectorXcf out_eigen = gEigenFFT.fwd(in_eigen);
            array_xc out(out_eigen.size(),false) ;
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.size()*sizeof(float) * 2);
            return out;
        }
        inline array_xf idft(const array_xc& in)
        {
            Eigen::Map<Eigen::VectorXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0));
            Eigen::VectorXf out_eigen = gEigenFFT.inv(in_eigen);
            array_xf out(out_eigen.size(),false) ;
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.size()*sizeof(float));
            return out;
        }
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_KOKKOSARRAY_FFTW" << std::endl;
            Eigen::Map<Eigen::ArrayXXf> in_eigen((float*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);
            auto out = gen_2d_view<array_xxc>(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);
            auto out = Zero<array_xxc>(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline void dft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

        }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
            auto out = Zero<array_xxc>(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

            return out;
        }
        inline void  idft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

        }
        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
            auto out = Zero<array_xxf>(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar));

            return out;
        }
        inline void idft_cr(const array_xxc& in, array_xxf& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar));
        }

    }  // namespace KokkosArray
}  // namespace WireCell

#endif
