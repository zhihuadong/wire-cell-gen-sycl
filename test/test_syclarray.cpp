#include "WireCellGenSycl/SyclEnv.h"
#include "WireCellGenSycl/SyclArray.h"
#include <omp.h>


using namespace WireCell::GenSycl ;

int main () {

double t0 = omp_get_wtime() ;
  const size_t N = 100000 ;
  auto e = SyclEnv() ;
  auto q = SyclEnv::get_queue() ;

  
  float a_host[N] ={ 0.0 } ;
  for (int i =0 ; i<10 ; i++) a_host[i] = (float) i ;

  WireCell::SyclArray::array_xf a_d(N, false) ;
  a_d.copy_from( a_host ) ;
  //q.memcpy(a_d.data() , a_host, N*sizeof(float) ).wait() ;
  
  //auto a_d_ptr = a_d.data() ;
  //

  auto a_s = a_d.a() ;
  q.wait() ;
double t1 = omp_get_wtime();
std::cout<<"before Kernel time: "<< t1-t0<<std::endl ;

  q.submit([&] (sycl::handler &h ) {
  sycl::stream out(1024, 128, h ) ;

  h.parallel_for(N, [=](auto i) {
       int ii = i.get_id() ;
       WireCell::SyclArray::array_xf a_d_k(a_s) ;

    //   if(ii < 10 ) out<<"a_d["<<ii<<"]= "<< a_d_k[ii] << cl::sycl::endl ;
   //    if(ii < 10 ) printf( " a_d[%d]=%f \n" , ii, a_d_k[ii] ) ;
       a_d_k(ii) += float(ii*ii) ;
      // if(ii < 10 ) printf( " a_d[%d]=%f \n" , ii, a_d[ii] ) ;
       //a_d[ii] += float(ii*ii) ;
		  } ) ;

  }) ;
  q.wait() ;
  auto rst = a_d.to_host() ;
  q.wait() ;
double t2 = omp_get_wtime() ;
std::cout<<"Kernel time: "<< t2-t1<<std::endl ;

  //float * rst = (float * ) malloc (sizeof(float) * N)   ;
  //   q.memcpy(rst, a_d_ptr, N* sizeof(float) ) ;

  std::cout <<"result: " ;
  for ( int i=0 ; i<100; i++ ) { 
    	std::cout<<rst[i] <<" " ;  
  }
  std::cout<<std::endl ;
  free(rst) ;
  return 0 ; 

}
