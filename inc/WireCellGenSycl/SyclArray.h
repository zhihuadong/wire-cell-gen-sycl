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
		public :
		    Array1D() { 
			    q_ = GenSycl::SyclEnv::get_queue() ;
			    ptr_ = NULL ; 
			    sz_ =0 ; 
		    } 
		    ~Array1D( ) { sycl::free( ptr_, q_ ) ; } ;
	            Array1D( T* ptr, size_t N ) {
			ptr_ = ptr ;
			sz_ = N ;

		    }
		    Array1D ( size_t N , bool init = true ) {
			q_ = GenSycl::SyclEnv::get_queue() ;
			ptr_ = sycl::malloc_device<T> ( N, q_ ) ;   
			if(init) {
		            q_.memset(ptr_, 0 , N*sizeof(T) )  ;
			}	
		    }
		    T & operator[]  (size_t i) const { return ptr_[i] ; }; 
		    T & operator[]  (size_t i)  { return ptr_[i] ; }; 
		    T &operator()(size_t i)const { return ptr_[i] ; };  
		    T &operator()(size_t i) { return ptr_[i] ; };  
		    void set( T value ) { 
			    auto ptr = ptr_ ;
			    q_.parallel_for( sz_ ,[=] (auto i ) {
				ptr[i] = value ;  
			    }).wait() ;
		    }
		    void copy_to( T* d_ptr) {
			    q_.memcpy(d_ptr , ptr_ , sz_* sizeof(T) ).wait() ;
		    }
		    void copy_from( T* s_ptr) {
			    q_.memcpy( ptr_ ,(void * ) s_ptr, sz_* sizeof(T) ).wait() ;
		    }
		    void copy_from( void * s_ptr) {
			    q_.memcpy( ptr_ ,  s_ptr, sz_* sizeof(T) ).wait() ;
		    }
		    T*  to_host() {
			    T* ret = (T*) malloc( sizeof(T) * sz_) ;
			    q_.memcpy( ret, ptr_,  sz_* sizeof(T) ).wait() ;
			    return ret ;
		    }
		    size_t extent( int i ) const  { 
			    if (i != 0 ){ 
				    std::cout<< "Array1D invalid dimmension: "<< i << std::endl ;
				   exit(1) ;
			    } else return sz_ ;
		    }
		    T * data() const { return ptr_ ; } 
		    void resize( size_t i ) { 
			    T* ptr1 = sycl::malloc_device<T> ( i , q_) ;
			    size_t j = sz_ > i ?  i : sz_ ;
			    q_memcpy(ptr1, ptr_ , j * sizeof (T) ).wait() ;
			    if ( j > sz_ ) q_memset(ptr1+sz_, 0 , (j-sz_) * sizeof(T) ) ;  
			    sycl::free (ptr_, q_) ;
			    ptr_ = ptr1 ;
			    sz_  = i ;
		    }
		    void free() { sycl::free(ptr_,q_) ; }
		    auto get_queue() {return q_ ; }  
		private :
		    T* ptr_ ;
		    size_t sz_ ;
		    cl::sycl::queue q_ ;

	} ;
        /// A  2D array
	template < class T >  class Array2D 
	{
		public :
		    Array2D() { 
			    q_ = GenSycl::SyclEnv::get_queue() ;
			    ptr_ = NULL ; 
			    sz1_ =0 ; 
			    sz2_ =0 ; 
		    } 
		    ~Array2D( ) { sycl::free( ptr_, q_ ) ; } ;
	            Array2D( T* ptr, size_t N, size_t M ) {
			ptr_ = ptr ;
			sz1_ = N ;
			sz2_ = M ;

		    }
		    Array2D ( size_t N , size_t M , bool init = true ) {
			q_ = GenSycl::SyclEnv::get_queue() ;
			ptr_ = sycl::malloc_device<T> ( N*M, q_ ) ;   
			if(init) {
		            q_.memset(ptr_, 0 , N*M*sizeof(T)) ;
			}	
		    }
		    T& operator()(size_t i, size_t j ) const { return ptr_[j*sz1_ +  i ] ; };  
		    void set( const T& value ) { 
			    auto ptr = ptr_ ;
			    auto v = value ;
			    q_.parallel_for( sz1_*sz2_ ,[=] (auto i ) {
				ptr[i] = v ;  
			    }).wait() ;
		    }
		    void copy_to( T* d_ptr) {
			    q_.memcpy(d_ptr , ptr_ , sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    void copy_from( T* s_ptr) {
			    q_.memcpy( ptr_ ,(void*) s_ptr, sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    void copy_from( void* s_ptr) {
			    q_.memcpy( ptr_ , s_ptr, sz1_*sz2_* sizeof(T) ).wait() ;
		    }
		    T*  to_host() {
			    T* ret = (T*) malloc( sizeof(T) * sz1_*sz2_) ;
			    q_.memcpy( ret, ptr_,  sz1_*sz2_ * sizeof(T) ).wait() ;
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
			    T* ptr1 = sycl::malloc_device<T> ( i*j  , q_) ;
			    auto sz1 = sz1_ ;
			    auto sz2 = sz2_ ;
			    auto ptr = ptr_ ;
			    q_.parallel_for( {i , j} , [=] ( auto item ) {
			        auto ii = item.get_id(0) ;
			        auto jj = item.get_id(1) ;
		                ptr1[ii + jj * i ] = (ii < sz1 && jj <sz2)   ? ptr[ ii + jj*sz1 ] :  value ;
			    }).wait() ;
			    sycl::free (ptr_,q_ ) ;
			    ptr_ = ptr1 ;
			    sz1_  = i ;
			    sz2_  = j ;
		    }
		    void free() { sycl::free(ptr_,q_) ; }
		    auto get_queue() {return q_ ; }  
		private :
		    T* ptr_ ;
		    size_t sz1_ ;
		    size_t sz2_ ;
		    cl::sycl::queue q_ ;

	}; 

	struct float_2 {
		float x ;
		float y ;

	} ;

	template < class T >  class WComplex  //try own with simple wc  needed method only 
	{	
		public :
			WComplex() { x_=0 ; y_=0 ; } 
			~WComplex() {} ;
			WComplex( T a , T b) { x_= a; y_= b; } 
                	WComplex(const T  a) {x_= a; y_ = 0.0 ; }
			WComplex(const WComplex & a ) {  
				x_ = a.Real() ;
				y_ = a.Imag() ; 
			}
			T Real()const  { return x_ ; } 
			T Imag()const  { return y_ ; }


			WComplex & operator+ ( const WComplex & a) {
				WComplex ret(x_+ a.Real(), y_ + a.Imag() ) ;
				return ret ;
			}	
			void operator *= (const WComplex & a ) {
				x_ = x_ * a.Real() - y_ * a.Imag() ;
				y_ = x_ * a.Imag() + y_ * a.Real() ;
			}
			void operator /= (const T& a ) {
				if(a == 0.0  ) { 
				   	printf( "divided by 0\n" )  ;
				} else  {
					x_ /= a ;
					y_ /= a ;
				}
			}
			WComplex & operator- ( const WComplex & a ) {
				 WComplex ret(x_- a.Real(), y_ - a.Imag() ) ; 
				 return ret ;
			}	

			void operator = ( const WComplex & a ) {
				x_ = a.Real() ;
				y_ = a.Imag() ;
			}


//		private :
			T x_ ;
			T y_ ;
	};
	
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
/*
        /// Dump out a string for pinting for a 2D view.
        template <class ViewType>
        inline std::string dump_2d_view(const ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << ", " << A.extent(1) << "} :\n";

            auto h_A = Kokkos::create_mirror_view(A);
            Kokkos::deep_copy(h_A, A);

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
                    ss << h_A(i, j) << " ";
                }
                ss << std::endl;
            }

            bool all_zero = true;
            for (Index i = 0; i < N0 && all_zero == true; ++i) {
                for (Index j = 0; j < N1 && all_zero == true; ++j) {
                    if (h_A(i, j) != 0) {
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
        inline std::string dump_1d_view(const ViewType& A, const Index length_limit = 20)
        {
            std::stringstream ss;
            ss << typeid(ViewType).name() << ", shape: {" << A.extent(0) << "} :\n";

            auto h_A = Kokkos::create_mirror_view(A);
            Kokkos::deep_copy(h_A, A);

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
                ss << h_A(j) << " ";
            }
            ss << std::endl;

            bool all_zero = true;
            for (Index j = 0; j < N0 && all_zero == true; ++j) {
                if (h_A(j) != 0) {
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


#if defined ENABLE_CUDA
    #include "WireCellGenKokkos/SyclArray_cuda.h"
#elif defined ENABLE_HIP
    #include "WireCellGenKokkos/SyclArray_hip.h"
#else
    #include "WireCellGenKokkos/SyclArray_fftw.h"
#endif
*/
    }  // namespace SyclArray
}  // namespace WireCell

#include "WireCellGenSycl/SyclArray_cuda.h"

#endif
