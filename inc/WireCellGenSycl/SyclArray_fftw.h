/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_SYCLARRAY_FFTW
#define WIRECELL_SYCLARRAY_FFTW

#include <iostream> //debug

#include <fftw3.h>

namespace WireCell {

namespace SyclArray {
inline array_xc dft(const array_xf& in)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_1d_r2c" << std::endl;
    int N = in.extent(0) ;
    array_xc out(N) ;
    fftwf_plan p = fftwf_plan_dft_r2c_1d(N, in.data(), reinterpret_cast<fftwf_complex *> (out.data()), FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    return out;
}
inline array_xf idft(const array_xc& in)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_1d_c2r" << std::endl;
    int N = in.extent(0) ;
    array_xf out(N) ;
    fftwf_plan p = fftwf_plan_dft_c2r_1d(N, reinterpret_cast<fftwf_complex *> (in.data()), out.data(), FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    auto q = out.get_queue() ;
    auto ptr = out.data() ;
    q.parallel_for(sycl::range<1>(N), [=] ( auto item) {
        auto i0 = item.get_id(0);
        ptr[i0] /= N ;
    } ) ;
    return out;
}
inline void dft_rc(const array_xxf& in, array_xxc& out,int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_r2c_1" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    fftwf_plan p ;
    if (dim == 0 ) {
        int n[] = {(int) N1};
        int inembed[] = {(int) N1} ;
        int onembed[] = {(int) N1} ;

        p = fftwf_plan_many_dft_r2c( 1, n, (int) N0,
                                     in.data(), inembed,
                                     (int) N0, 1,
                                     reinterpret_cast<fftwf_complex *> ( out.data()), onembed,
                                     (int) N0, 1,
                                     FFTW_ESTIMATE );
    }
    if (dim == 1 ) {
        int n[] = {(int) N0};
        int inembed[] = {(int) N0} ;
        int onembed[] = {(int) N0} ;

        p = fftwf_plan_many_dft_r2c( 1, n, (int) N1,
                                     in.data(), inembed,
                                     1, (int) N0,
                                     reinterpret_cast<fftwf_complex *> (out.data()), onembed,
                                     1, (int) N0,
                                     FFTW_ESTIMATE );
    }
    fftwf_execute(p) ;
    fftwf_destroy_plan(p);
}
inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_r2c_0" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    auto  out = array_xxc( N0, N1);
    dft_rc(in, out, dim) ;
    return out;
}
inline void dft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_c2c_1" <<" "<<dim<<" "<< std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    fftwf_plan p ;
    if (dim == 0 ) {
        int n[] = {(int) N1};
        int inembed[] = {(int) N1} ;
        int onembed[] = {(int) N1} ;

        p = fftwf_plan_many_dft( 1, n, (int) N0,
                                 reinterpret_cast<fftwf_complex* > ( in.data()), inembed,
                                 (int) N0, 1,
                                 reinterpret_cast<fftwf_complex *> ( out.data()), onembed,
                                 (int) N0, 1,
                                 FFTW_FORWARD, FFTW_ESTIMATE );
    }
    if (dim == 1 ) {
        int n[] = {(int) N0};
        int inembed[] = {(int) N0} ;
        int onembed[] = {(int) N0} ;

        p = fftwf_plan_many_dft( 1, n, (int) N1,
                                 reinterpret_cast<fftwf_complex *> (in.data()), inembed,
                                 1, (int) N0,
                                 reinterpret_cast<fftwf_complex *> (out.data()), onembed,
                                 1, (int) N0,
                                 FFTW_FORWARD, FFTW_ESTIMATE );
    }
    fftwf_execute(p) ;
    fftwf_destroy_plan(p);
}
inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_c2c_0" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    auto  out = array_xxc( N0, N1);
    dft_cc(in, out, dim) ;
    return out;
}
inline void  idft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_I_c2c_1" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    fftwf_plan p ;

    auto q = out.get_queue() ;
    auto ptr= out.data() ;

    if (dim == 0 ) {
        int n[] = {(int) N1};
        int inembed[] = {(int) N1} ;
        int onembed[] = {(int) N1} ;

        p = fftwf_plan_many_dft( 1, n, (int) N0,
                                 reinterpret_cast<fftwf_complex* > ( in.data()), inembed,
                                 (int) N0, 1,
                                 reinterpret_cast<fftwf_complex *> ( out.data()), onembed,
                                 (int) N0, 1,
                                 FFTW_BACKWARD, FFTW_ESTIMATE );
        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto item) {
            auto i0 = item.get_id(0) ;
            ptr[i0].x /= N1 ;
            ptr[i0].y /=N1 ;
        } ) ;
    }
    if (dim == 1 ) {
        int n[] = {(int) N0};
        int inembed[] = {(int) N0} ;
        int onembed[] = {(int) N0} ;

        p = fftwf_plan_many_dft( 1, n, (int) N1,
                                 reinterpret_cast<fftwf_complex *> (in.data()), inembed,
                                 1, (int) N0,
                                 reinterpret_cast<fftwf_complex *> (out.data()), onembed,
                                 1, (int) N0,
                                 FFTW_BACKWARD, FFTW_ESTIMATE );
        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto item) {
            auto i0 = item.get_id(0) ;
            ptr[i0].x /= N0 ;
            ptr[i0].y /=N0 ;
        } ) ;
    }
    fftwf_execute(p) ;
    fftwf_destroy_plan(p);

}
inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_I_c2c_0" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    auto  out = array_xxc( N0, N1);
    idft_cc(in, out, dim) ;
    return out;
}
inline void idft_cr(const array_xxc& in, array_xxf& out, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_I_c2r_1" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    fftwf_plan p ;

    auto q = out.get_queue() ;
    auto ptr= out.data() ;

    if (dim == 0 ) {
        int n[] = {(int) N1};
        int inembed[] = {(int) N1} ;
        int onembed[] = {(int) N1} ;

        p = fftwf_plan_many_dft_c2r( 1, n, (int) N0,
                                     reinterpret_cast<fftwf_complex *> ( in.data()), inembed,
                                     (int) N0, 1,
                                     out.data(), onembed,
                                     (int) N0, 1,
                                     FFTW_ESTIMATE );
        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto item) {
            auto i0 = item.get_id(0) ;
            ptr[i0] /= N1 ;
        } ) ;
    }
    if (dim == 1 ) {
        int n[] = {(int) N0};
        int inembed[] = {(int) N0} ;
        int onembed[] = {(int) N0} ;

        p = fftwf_plan_many_dft_c2r( 1, n, (int) N1,
                                     reinterpret_cast<fftwf_complex *> (in.data()), inembed,
                                     1, (int) N0,
                                     out.data(), onembed,
                                     1, (int) N0,
                                     FFTW_ESTIMATE );
        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto item) {
            auto i0 = item.get_id(0) ;
            ptr[i0] /= N0 ;
        } ) ;
    }
    fftwf_execute(p) ;
    fftwf_destroy_plan(p);
}
inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
{
    std::cout << "WIRECELL_SYCLARRAY_FFTW_2d_I_c2r_0" << std::endl;
    Index N0 = in.extent(0);
    Index N1 = in.extent(1);
    auto  out = array_xxf( N0, N1);
    idft_cr(in, out, dim) ;
    return out;
}

}  // namespace SyclArray
}  // namespace WireCell

#endif
