/**
 * Similar like the WireCell::Array with Eigen backend,
 * this KokkosArray provides interface for FFTs.
 */

#ifndef WIRECELL_SYCLARRAY
#define WIRECELL_SYCLARRAY

#include <CL/sycl.hpp>
#include "SyclEnv.h"
#include <string>
#include <typeinfo>

//using namespace sycl ;
namespace WireCell {

    namespace SyclArray {
        using Scalar = float;
        using Index = int;

        /// A  1D array
	template < class T > 
	class Array1D 
	{
	        struct A1d_s { 
			T* ptr; 
			size_t sz ; 
		}  ;  
		public :
		    Array1D() { 
		//	    q_ = GenSycl::SyclEnv::get_queue() ;
			    a_.ptr = NULL ; 
			    a_.sz =0 ; 
		    } 
		    //~Array1D( ) { sycl::free( ptr_, q_ ) ; } ;
		    ~Array1D( ) {  } ;
	            Array1D( T* ptr, size_t N ) {
			a_.ptr = ptr ;
			a_.sz = N ;
		    }
	            Array1D( struct A1d_s a ) {
			a_.ptr = a.ptr ;
			a_.sz = a.sz ;
		    }
	            auto  a() {
			    return a_ ;
		    }
		    void reset(){
			    auto q = GenSycl::SyclEnv::get_queue();  
			    sycl::free(a_.ptr, q) ; 
			    a_.ptr=NULL ; 
			    a_.sz=0 ;
		    }; 
		    Array1D ( size_t N , bool init = true ) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			a_.ptr = sycl::malloc_device<T> ( N, q ) ;   
			if(init) {
		            q.memset(a_.ptr, 0 , N*sizeof(T) )  ;
			}	
			a_.sz = N ;
		    }
		    T & operator[]  (size_t i) const { return a_.ptr[i] ; }; 
		    T & operator[]  (size_t i)  { return a_.ptr[i] ; }; 
		    T &operator()(size_t i)const { return a_.ptr[i] ; };  
		    T &operator()(size_t i) { return a_.ptr[i] ; };  
		    void set( T value ) { 
			    auto ptr = a_.ptr ;
			    auto size = a_.sz ;
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    q.parallel_for( size ,[=] (auto i ) {
				ptr[i] = value ;  
			    }).wait() ;
		    }
		    void copy_to( T* d_ptr) {
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy(d_ptr , a_.ptr , a_.sz* sizeof(T) ).wait() ;
		    }
		    void copy_from( T* s_ptr) {
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy( (void * )a_.ptr ,(void * ) s_ptr, a_.sz* sizeof(T) ).wait() ;
		    }
		    void copy_from( void * s_ptr) {
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy( a_.ptr,  s_ptr, a_.sz* sizeof(T) ).wait() ;
		    }
		    T*  to_host()  {
			    T* ret = (T*) malloc( sizeof(T) * a_.sz) ;
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy( (void *)ret, (void *)a_.ptr, a_.sz* sizeof(T) ).wait() ;
			    return ret ;
		    }
		    size_t extent( int i ) const  { 
			    if (i != 0 ){ 
				    std::cout<< "Array1D invalid dimmension: "<< i << std::endl ;
				   exit(1) ;
			    } else return a_.sz ;
		    }
		    T * data() const { return a_.ptr ; } 
		    void resize( size_t i ) { 
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    T* ptr1 = sycl::malloc_device<T> ( i , q) ;
			    size_t j = a_.sz > i ?  i : a_.sz ;
			    q.memcpy(ptr1, a_.ptr , j * sizeof (T) ).wait() ;
			    if ( j > a_.sz ) q_memset(ptr1+a_.sz, 0 , (j-a_.sz) * sizeof(T) ) ;  
			    sycl::free (a_.ptr, q) ;
			    a_.ptr = ptr1 ;
			    a_.sz_  = i ;
		    }
		    void free() { 
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    sycl::free(a_.ptr,q) ;
		    }
		    auto get_queue() {
			    auto q = GenSycl::SyclEnv::get_queue() ;
			    return q ;
		    }  


		private :
		    struct A1d_s a_ ;

	} ;
        /// A  2D array
	template < class T >  class Array2D 
	{
		public :
		    Array2D() { 
			    ptr_ = NULL ; 
			    sz1_ =0 ; 
			    sz2_ =0 ; 
		    } 
		    ~Array2D( ) {
		//	   sycl::free( ptr_, q_ ) ; 
		    } ;
	            Array2D( T* ptr, size_t N, size_t M ) {
			ptr_ = ptr ;
			sz1_ = N ;
			sz2_ = M ;

		    }
		    Array2D ( size_t N , size_t M , bool init = true ) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			ptr_ = sycl::malloc_device<T> ( N*M, q ) ;   
			if(init) {
		            q.memset(ptr_, 0 , N*M*sizeof(T)) ;
			}	
			sz1_ = N ; 
			sz2_ = M ;
		    }
		    Array2D( Array2D & ar) {
			    ptr_ = ar.data() ;
			    sz1_ = ar.extent(0) ;
			    sz2_ = ar.extent(1) ;
		    }
		    void alloc(size_t N, size_t M) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    ptr_ = sycl::malloc_device<T> ( N*M, q ) ;
			    sz1_ = N ;
			    sz2_ = M ;
		    }

		    T& operator()(size_t i, size_t j ) const { return ptr_[j*sz1_ +  i ] ; };  
		    void set( const T value ) { 
			auto q = GenSycl::SyclEnv::get_queue() ;
			    auto ptr = ptr_ ;
			    auto v = value ;
			    q.parallel_for( sz1_*sz2_ ,[=] (auto i ) {
				ptr[i] = v ;  
			    }).wait() ;
		    }
		    void copy_to( T* d_ptr) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy(d_ptr , ptr_ , sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    void copy_from( T* s_ptr) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy( ptr_ ,(void*) s_ptr, sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    void copy_from( void* s_ptr) {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    q.memcpy( ptr_ , s_ptr, sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    T*  to_host()  {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    T* ret = (T*) malloc( sizeof(T) * sz1_*sz2_) ;
			    q.memcpy( ret, ptr_,  sz1_*sz2_ * sizeof(T) ).wait() ;
			    return ret ;
		    }

		    size_t extent( const int i ) const { 
			    if (i == 0  ){ return sz1_ ; }
			    else if( i == 1 ) {return sz2_ ; } 
			    else { 
				  std::cout<< "Array2D invalid dimmension: "<< i << std::endl ;
				  exit(1) ;
			    } 
		    }
		    T * data() const { return ptr_ ; } 
		    void resize( size_t i, size_t j, T value ) { 
			auto q = GenSycl::SyclEnv::get_queue() ;
			    T* ptr1 = sycl::malloc_device<T> ( i*j  , q) ;
			    auto sz1 = sz1_ ;
			    auto sz2 = sz2_ ;
			    auto ptr = ptr_ ;
			    q.parallel_for( {i , j} , [=] ( auto item ) {
			        auto ii = item.get_id(0) ;
			        auto jj = item.get_id(1) ;
		                ptr1[ii + jj * i ] = (ii < sz1 && jj <sz2)   ? ptr[ ii + jj*sz1 ] :  value ;
			    }).wait() ;
			    sycl::free (ptr_,q ) ;
			    ptr_ = ptr1 ;
			    sz1_  = i ;
			    sz2_  = j ;
		    }
		    void free() {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    sycl::free(ptr_,q) ; 
		    }
		    auto get_queue() {
			auto q = GenSycl::SyclEnv::get_queue() ;
			    return q ;
		    }  
		private :
		    T* ptr_ ;
		    size_t sz1_ ;
		    size_t sz2_ ;

	}; 

	struct float_2 {
		float x ;
		float y ;
		friend std::ostream& operator<<(std::ostream& os, const float_2& c ) {
			os<<"("<<c.x<<","<<c.y<<")" ;
			return os ;

		}	

	} ;

	typedef Array1D<Scalar> array_xf;

        /// A complex, 1D array
        typedef Array1D< float_2 > array_xc;

        /// A real, 2D array
        typedef Array2D<Scalar > array_xxf;

        /// A complex, 2D array
        typedef Array2D < float_2 > array_xxc;

	// Generate a 1D view initialized with given value.
        template <class ArrayType>
        inline  ArrayType  gen_1d_Array(const Index N0, const Scalar val = 0)
        {
            ArrayType ret(N0, false) ;
            ret.set( val ) ;
            return ret;
        }


        /// Generate a 2D view initialized with given value.
        template <class ArrayType>
        inline ArrayType gen_2d_Array(const Index N0, const Index N1, const Scalar val = 0)
        {
            ArrayType ret(N0, N1, false) ;
            ret.set( val ) ;
            return ret;
        }
        template <class ArrayType>
        inline ArrayType Zero(const Index N0, const Index N1)
        {
	    ArrayType ret(N0, N1, true) ;
	    return ret ;
        }

        /// Dump out a string for pinting for a 2D view.
        template <class ViewType>
        inline std::string dump_2d_view( ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << ", " << A.extent(1) << "} :\n";

            auto h_A = A.to_host() ; 

            Index N0 = A.extent(0);
            Index N1 = A.extent(1);
            bool print_dot0 = true;
            for (Index i = 0; i < N0; ++i) {
                if (i > length_limit && i < N0 - length_limit) {
                    if (print_dot0) {
                        ss << "... \n";
                        print_dot0 = false;
                    }
                    continue;
                }

                bool print_dot1 = true;
                for (Index j = 0; j < N1; ++j) {
                    if (j > length_limit && j < N1 - length_limit) {
                        if (print_dot1) {
                            ss << "... ";
                            print_dot1 = false;
                        }

                        continue;
                    }
                    //ss << h_A(i, j) << " ";
                    ss << h_A[i + j* N0 ] << " ";
                }
                ss << std::endl;
            }

            bool all_zero = true;
            for (Index i = 0; i < N0 && all_zero == true; ++i) {
                for (Index j = 0; j < N1 && all_zero == true; ++j) {
                    if (h_A[i + j* N0 ] != 0) {
                        all_zero = false;
                        break;
                    }
                }
            }
            if (all_zero) {
                ss << "All Zero!\n";
            }

            return ss.str();
        }

        inline std::string dump_2d_c( array_xxc& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(array_xxc).name() << ", shape: {" << A.extent(0) << ", " << A.extent(1) << "} :\n";

            auto h_A = A.to_host() ; 

            Index N0 = A.extent(0);
            Index N1 = A.extent(1);
            bool print_dot0 = true;
            for (Index i = 0; i < N0; ++i) {
                if (i > length_limit && i < N0 - length_limit) {
                    if (print_dot0) {
                        ss << "... \n";
                        print_dot0 = false;
                    }
                    continue;
                }

                bool print_dot1 = true;
                for (Index j = 0; j < N1; ++j) {
                    if (j > length_limit && j < N1 - length_limit) {
                        if (print_dot1) {
                            ss << "... ";
                            print_dot1 = false;
                        }

                        continue;
                    }
                    //ss << h_A(i, j) << " ";
                    ss << h_A[i + j* N0 ] << " ";
                }
                ss << std::endl;
            }

            bool all_zero = true;
            for (Index i = 0; i < N0 && all_zero == true; ++i) {
                for (Index j = 0; j < N1 && all_zero == true; ++j) {
                    if (h_A[i + j* N0 ].x != 0 && h_A[i + j* N0 ].y != 0 ) {
                        all_zero = false;
                        break;
                    }
                }
            }
            if (all_zero) {
                ss << "All Zero!\n";
            }

            return ss.str();
        }

        /// Dump out a string for pinting for a 1D view.
        template <class ViewType>
        inline std::string dump_1d_view( ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << "} :\n";

            auto h_A = A.to_host() ;

            Index N0 = A.extent(0);
            bool print_dot1 = true;
            for (Index j = 0; j < N0; ++j) {
                if (j > length_limit && j < N0 - length_limit) {
                    if (print_dot1) {
                        ss << "... ";
                        print_dot1 = false;
                    }
                    continue;
                }
                ss << h_A[j] << " ";
            }
            ss << std::endl;

            bool all_zero = true;
            for (Index j = 0; j < N0 && all_zero == true; ++j) {
                if (h_A[j] != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                ss << "All Zero!\n";
            }

            return ss.str();
        }

    }  // namespace KokkosArray
}  // namespace WireCell

/*
#if defined ENABLE_CUDA
    #include "WireCellGenKokkos/SyclArray_cuda.h"
#elif defined ENABLE_HIP
    #include "WireCellGenKokkos/SyclArray_hip.h"
#else
    #include "WireCellGenKokkos/SyclArray_fftw.h"
#endif

}  // namespace SyclArray
}  // namespace WireCell

*/
#include "WireCellGenSycl/SyclArray_cuda.h"

#endif
