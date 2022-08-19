/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_SYCLARRAY_FFTW
#define WIRECELL_SYCLARRAY_FFTW

#include <unsupported/Eigen/FFT>
#include <iostream> //debug

namespace WireCell {

    namespace SyclArray {
        /**
         * using eigen based WireCell::Array for fast prototyping
         * this may introduce extra data copying, will investigate later
         * by default eigen is left layout (column major)
         * https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
         * FIXME: float should be Scalar
         */
        thread_local static Eigen::FFT<float> gEigenFFT;
        thread_local static Eigen::FFT<float> gEigenFFT_dft_1d;      // c2c fwd and inv
        thread_local static Eigen::FFT<float> gEigenFFT_dft_r2c_1d;  // r2c fwd
        thread_local static Eigen::FFT<float> gEigenFFT_dft_c2r_1d;  // c2r inv

        inline array_xc dft(const array_xf& in)
        {
            Eigen::Map<Eigen::VectorXf> in_eigen((float*) in.data(), in.extent(0));
            Eigen::VectorXcf out_eigen = gEigenFFT.fwd(in_eigen);
            auto out = array_xc( out_eigen.size(),false) ;
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.size()*sizeof(float) * 2);
            return out;
        }
        inline array_xf idft(const array_xc& in)
        {
            Eigen::Map<Eigen::VectorXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0));
            Eigen::VectorXf out_eigen = gEigenFFT.inv(in_eigen);
            auto out = array_xf( out_eigen.size(),false) ;
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.size()*sizeof(float));
            return out;
        }
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_SYCLARRAY_FFTW" << std::endl;
            Eigen::Map<Eigen::ArrayXXf> in_eigen((float*) in.data(), in.extent(0), in.extent(1));
            
            // auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);

            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            Eigen::MatrixXcf out_eigen(nrows, ncols);

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf fspec(ncols);
                    Eigen::VectorXf tmp = in_eigen.row(irow);
                    gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
                    out_eigen.row(irow) = fspec;
                }
            }
            else if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf fspec(nrows);
                    Eigen::VectorXf tmp = in_eigen.col(icol);
                    gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
                    out_eigen.col(icol) = fspec;
                }
            }

            auto out = array_xxc(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);
	    std::cout<<"fftw:dft_rc:"<<std::endl ;

            return out;
        }
        inline void dft_rc(const array_xxf& in, array_xxc& out, int dim = 0)
        {
            std::cout << "WIRECELL_SYCLARRAY_FFTW" << std::endl;
            Eigen::Map<Eigen::ArrayXXf> in_eigen((float*) in.data(), in.extent(0), in.extent(1));
            
            // auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);

            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            Eigen::MatrixXcf out_eigen(nrows, ncols);

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf fspec(ncols);
                    Eigen::VectorXf tmp = in_eigen.row(irow);
                    gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
                    out_eigen.row(irow) = fspec;
                }
            }
            else if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf fspec(nrows);
                    Eigen::VectorXf tmp = in_eigen.col(icol);
                    gEigenFFT_dft_r2c_1d.fwd(fspec, tmp);  // r2c
                    out_eigen.col(icol) = fspec;
                }
            }
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);
	    std::cout<<"fftw:dft_rc:"<<std::endl ;
        }
         inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));

            // auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);

            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            Eigen::MatrixXcf out_eigen(nrows, ncols);

            out_eigen = in_eigen.matrix();

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf pspec(ncols);
                    gEigenFFT_dft_1d.fwd(pspec, out_eigen.row(irow));  // c2c
                    out_eigen.row(irow) = pspec;
                }
            }
            else {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf pspec(nrows);
                    gEigenFFT_dft_1d.fwd(pspec, out_eigen.col(icol));  // c2c
                    out_eigen.col(icol) = pspec;
                }
            }

            auto out = array_xxc(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

	    std::cout<<"fftw:dft_cc:"<<std::endl ;
            return out;
        }

        inline void dft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));

            // auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);

            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            Eigen::MatrixXcf out_eigen(nrows, ncols);

            out_eigen = in_eigen.matrix();

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf pspec(ncols);
                    gEigenFFT_dft_1d.fwd(pspec, out_eigen.row(irow));  // c2c
                    out_eigen.row(irow) = pspec;
                }
            }
            else {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf pspec(nrows);
                    gEigenFFT_dft_1d.fwd(pspec, out_eigen.col(icol));  // c2c
                    out_eigen.col(icol) = pspec;
                }
            }
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);
	    std::cout<<"fftw:dft_cc:"<<std::endl ;

        }

        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            // auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
            
            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            // gEigenFFT works on matrices, not arrays, also don't step on const input
            Eigen::MatrixXcf out_eigen(nrows, ncols);
            out_eigen = in_eigen.matrix();

            if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf pspec(nrows);
                    gEigenFFT_dft_1d.inv(pspec, out_eigen.col(icol));  // c2c
                    out_eigen.col(icol) = pspec;
                }
            }
            else if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf pspec(ncols);
                    gEigenFFT_dft_1d.inv(pspec, out_eigen.row(irow));  // c2c
                    out_eigen.row(irow) = pspec;
                }
            }
            auto out = array_xxc(out_eigen.rows(), out_eigen.cols());
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);

	    std::cout<<"fftw:idft_cc:"<<std::endl ;
            return out;
        }

        inline void  idft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            // auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
            
            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            // gEigenFFT works on matrices, not arrays, also don't step on const input
            Eigen::MatrixXcf out_eigen(nrows, ncols);
            out_eigen = in_eigen.matrix();

            if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXcf pspec(nrows);
                    gEigenFFT_dft_1d.inv(pspec, out_eigen.col(icol));  // c2c
                    out_eigen.col(icol) = pspec;
                }
            }
            else if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXcf pspec(ncols);
                    gEigenFFT_dft_1d.inv(pspec, out_eigen.row(irow));  // c2c
                    out_eigen.row(irow) = pspec;
                }
            }
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar) * 2);
	    std::cout<<"fftw:idft_cc:"<<std::endl ;

        }

        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            // auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            // gEigenFFT works on matrices, not arrays, also don't step on const input
            Eigen::MatrixXcf partial(nrows, ncols);
            partial = in_eigen.matrix();

            Eigen::ArrayXXf out_eigen(nrows, ncols);

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXf wave(ncols);                        // back to real-valued time series
                    gEigenFFT_dft_c2r_1d.inv(wave, partial.row(irow));  // c2r
                    out_eigen.row(irow) = wave;
                }
            }
            else if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXf wave(nrows);
                    gEigenFFT_dft_c2r_1d.inv(wave, partial.col(icol));  // c2r
                    out_eigen.col(icol) = wave;
                }
            }
            auto out = array_xxf(out_eigen.rows(), out_eigen.cols(), 0);
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar));
	    std::cout<<"fftw:idft_cr:"<<std::endl ;
            out_eigen.resize(0, 0) ;
            return out;
        }
        inline void idft_cr(const array_xxc& in, array_xxf& out, int dim = 0)
        {
            Eigen::Map<Eigen::ArrayXXcf> in_eigen((std::complex<float>*) in.data(), in.extent(0), in.extent(1));
            // auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
            const int nrows = in_eigen.rows();
            const int ncols = in_eigen.cols();

            // gEigenFFT works on matrices, not arrays, also don't step on const input
            Eigen::MatrixXcf partial(nrows, ncols);
            partial = in_eigen.matrix();

            Eigen::ArrayXXf out_eigen(nrows, ncols);

            if (dim == 0) {
                for (int irow = 0; irow < nrows; ++irow) {
                    Eigen::VectorXf wave(ncols);                        // back to real-valued time series
                    gEigenFFT_dft_c2r_1d.inv(wave, partial.row(irow));  // c2r
                    out_eigen.row(irow) = wave;
                }
            }
            else if (dim == 1) {
                for (int icol = 0; icol < ncols; ++icol) {
                    Eigen::VectorXf wave(nrows);
                    gEigenFFT_dft_c2r_1d.inv(wave, partial.col(icol));  // c2r
                    out_eigen.col(icol) = wave;
                }
            }
            memcpy( (void*)out.data(), (void*)out_eigen.data(), out_eigen.rows()*out_eigen.cols()*sizeof(Scalar));
	    std::cout<<"fftw:idft_cr:"<<std::endl ;
            out_eigen.resize(0, 0) ;

        }

    }  // namespace SyclArray
}  // namespace WireCell

#endif
