#ifndef WIRECELL_GENSYCL_SYCLENV
#define WIRECELL_GENSYCL_SYCLENV

#include <CL/sycl.hpp>

namespace WireCell {
    namespace GenSycl {
     //   class SyclEnv : public WireCell::ITerminal {
        class SyclEnv  {
           private:
            static bool Sycl_initialized;
	    static cl::sycl::context* ctx_ ;
	    static cl::sycl::device  device_ ;		    
            static cl::sycl::queue   queue_ ;

           public:
            SyclEnv();
            virtual ~SyclEnv();
	    bool Init() ;
	    static cl::sycl::context* get_ctx() { return ctx_ ; } ;
	    static cl::sycl::queue get_queue() { return queue_ ; } ;
	    static cl::sycl::device get_device() { return device_ ; } ;
            virtual void finalize();
        };

    }  // namespace GenSycl
}  // namespace WireCell

#endif
