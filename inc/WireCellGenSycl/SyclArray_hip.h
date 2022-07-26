/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_SYCLARRAY_HIP
#define WIRECELL_SYCLARRAY_HIP

#ifdef __CUDA_ARCH__
#define __CUDA_PASS__
#undef __CUDA_ARCH__
#endif

#include <hipfft.h>

#ifdef __CUDA_PASS__
#undef __CUDA_PASS__
#define __CUDA_ARCH__
#endif


#include <assert.h>

namespace WireCell {

    namespace SyclArray {
        inline array_xc dft(const array_xf& in)
        {
            Index N0 = in.extent(0);
            //auto out = gen_1d_Array<array_xc>(N0, 0);
            //auto out = array_xc(N0, false);
            auto out = array_xc(N0 );
	    hipfftHandle plan = NULL;
	    hipfftResult status ;
	    status = hipfftPlan1d( &plan, N0, HIPFFT_R2C, 1) ;
	    assert(status == HIPFFT_SUCCESS) ;
	    status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
	    assert(status == HIPFFT_SUCCESS) ;
            hipfftDestroy(plan) ;
            return out;
        }
        inline array_xf idft(const array_xc& in)
        {
            Index N0 = in.extent(0);
            auto out = array_xf(N0);
	    hipfftHandle plan = NULL;
            hipfftResult status ;
            status = hipfftPlan1d( &plan, N0, HIPFFT_C2R, 1) ;
            assert(status == HIPFFT_SUCCESS) ;
            status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
            assert(status == HIPFFT_SUCCESS) ;
            hipfftDestroy(plan) ;
	    auto q = out.get_queue() ;
	    auto ptr = out.data() ;
	    q.parallel_for(sycl::range<1> (N0), [=] ( auto ii0) { auto i0 = ii0.get_id(0) ; ptr[i0] /= N0 ; } ).wait() ;
            return out;
        }
        inline array_xxc dft_rc(const array_xxf& in, int dim = 0)
        {
            std::cout << "WIRECELL_SYCLARRAY_HIP" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            auto out = array_xxc(N0, N1);
	    //std::cout<<"(N0,N1)="<<N0<<","<<N1<<std::endl ;

	    hipfftHandle plan = NULL;

            hipfftResult status ;
            hipError_t sync_status ;


	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_R2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 , (int) N0, onembed, 1, N0, HIPFFT_R2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }
            return out;
        }
        inline  void dft_rc(const array_xxf& in, array_xxc& out, int dim = 0)
        {
            std::cout << "WIRECELL_SYCLARRAY_HIP" << std::endl;
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
	    


            hipfftHandle plan = NULL;

            hipfftResult status ;
            hipError_t sync_status ;


	    if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_R2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 , (int) N0, onembed, 1, N0, HIPFFT_R2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecR2C( plan,  (float * )in.data(), (float2 *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

        }
        inline array_xxc dft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            //auto out = gen_2d_Array<array_xxc>(N0, N1, 0);
            auto out = array_xxc(N0, N1);

	    hipfftHandle plan ;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (hipfftComplex * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed,  1, N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (hipfftComplex  * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }

           return out;
        }
        inline void dft_cc(const array_xxc& in, array_xxc& out, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);

            hipfftHandle plan ;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (hipfftComplex * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed,  1, N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (hipfftComplex  * )in.data(), (hipfftComplex *) out.data() , HIPFFT_FORWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

            }
       }
        inline array_xxc idft_cc(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            //auto out = gen_2d_Array<array_xxc>(N0, N1, 0);
            auto out = array_xxc(N0, N1);
	    auto q = out.get_queue() ;
	     hipfftHandle plan ;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data(), HIPFFT_BACKWARD ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
	        auto ptr = out.data() ;
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0].x /= N1 ; ptr[i0].y /=N1 ; } ).wait() ;
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 ,N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data() ,HIPFFT_BACKWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
	        auto ptr = out.data() ;
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0].x /= N0 ; ptr[i0].y /=N0 ; } ).wait() ;
            }

           return out;
        }
        inline void  idft_cc(const array_xxc& in,  array_xxc& out, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);

	    auto q = out.get_queue() ;
	    hipfftHandle plan ;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
                int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2C, (int) N0, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data(), HIPFFT_BACKWARD ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
	        auto ptr = out.data() ;
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0].x /= N1 ; ptr[i0].y /=N1 ; } ).wait();
            }

            if (dim == 1) {
                int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1 ,N0, onembed, 1, N0, HIPFFT_C2C, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2C( plan,  (float2 * )in.data(), (float2 *) out.data() ,HIPFFT_BACKWARD) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
	        auto ptr = out.data() ;
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0].x /= N0 ; ptr[i0].y /=N0 ; } ).wait() ;
            }

        }

        inline array_xxf idft_cr(const array_xxc& in, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);
            //auto out = gen_2d_Array<array_xxf>(N0, N1, 0);
            auto out = array_xxf(N0, N1);
	    auto q = out.get_queue() ;
	    auto ptr = out.data() ;
		
	    hipfftHandle plan = NULL;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
		int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2R, (int) N0, &worksize);
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
    
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0]/= N1 ;  } ).wait() ;
            }

            if (dim == 1) {
		int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1, N0, onembed, 1, N0, HIPFFT_C2R, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0]/= N0 ; } ).wait() ;
            }

            return out;
        }

        inline void idft_cr(const array_xxc& in, array_xxf& out, int dim = 0)
        {
            Index N0 = in.extent(0);
            Index N1 = in.extent(1);

	    auto q = out.get_queue() ;
	    auto ptr = out.data() ;

       	    hipfftHandle plan = NULL;

            hipfftResult status ;
            hipError_t sync_status ;

            if (dim == 0) {
		int n[] = { (int) N1};
                int inembed[] = {(int) N1};
                int onembed[] = {(int) N1};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, (int) N0, 1, onembed, N0, 1, HIPFFT_C2R, (int) N0, &worksize);
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);
    
	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0] /= N1  ; } ).wait() ;
            }

            if (dim == 1) {
		int n[] = {(int) N0};
                int inembed[] = {(int) N0};
                int onembed[] = {(int) N0};
                size_t worksize = 0;
                status = hipfftCreate(&plan);
                assert(status == HIPFFT_SUCCESS) ;
                // set to autoAllocate
                status = hipfftSetAutoAllocation( plan, 1);
                assert(status == HIPFFT_SUCCESS) ;

                //MakePlan
                status = hipfftMakePlanMany( plan, 1, n, inembed, 1, N0, onembed, 1, N0, HIPFFT_C2R, (int) N1, &worksize);
                //std::cout<<"worksize= "<<worksize << std::endl ;
                assert(status == HIPFFT_SUCCESS) ;

                //Excute
                status = hipfftExecC2R( plan,  (float2 * )in.data(), (float *) out.data() ) ;
                assert(status == HIPFFT_SUCCESS) ;

                // Wait for execution to finish
                sync_status=hipDeviceSynchronize() ;

                //Destroy plan
                hipfftDestroy(plan);

	        q.parallel_for(sycl::range<1>(N0*N1), [=] ( auto ii0) {  auto i0 = ii0.get_id(0); ptr[i0] /= N0 ;  } ).wait() ;
            }

        }

    }  // namespace SyclArray
}  // namespace WireCell

#endif
