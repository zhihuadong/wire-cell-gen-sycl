#include <CL/sycl.hpp>

#include "WireCellGenSycl/SyclEnv.h"
#include "WireCellUtil/Exceptions.h"
//#include "WireCellUtil/NamedFactory.h"
#include "WireCellGenSycl/syclcommon.h"
#include <iostream>

//WIRECELL_FACTORY(SyclEnv, WireCell::GenSycl::SyclEnv, WireCell::ITerminal)

using namespace WireCell;
using namespace std;

bool GenSycl::SyclEnv::Sycl_initialized = false;
cl::sycl::context* GenSycl::SyclEnv::ctx_  ;
cl::sycl::queue GenSycl::SyclEnv::queue_  ;
cl::sycl::device GenSycl::SyclEnv::device_  ;


GenSycl::SyclEnv::SyclEnv()
{
    std::cout << "Sycl::initializing" << std::endl;
    if (!Sycl_initialized) {
        try {
            Sycl_initialized = Init();
        }
        catch (Exception& e) {
            cerr << errstr(e) << endl;
            THROW(RuntimeError() << errmsg{"Sycl::initialize() FAILED!"});
        }
    }
    std::cout << "Sycl::initialized" << std::endl;
}

GenSycl::SyclEnv::~SyclEnv() {}

bool inline GenSycl::SyclEnv::Init() {
    if(Sycl_initialized ) {
	    return true ;
	}
    
    /*
    exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
        }
     } 
                  };
    */
     // Initialize device, queue and context
     if (!ctx_) {
        device_ = GenSycl::syclcommon::GetTargetDevice();

        queue_ = cl::sycl::queue(device_);
        ctx_ = new cl::sycl::context(queue_.get_context());
      } else {
        device_ = ctx_->get_devices()[0];
        queue_ = cl::sycl::queue(*ctx_, device_);
      }
	     return true ;

}



void GenSycl::SyclEnv::finalize()
{
    std::cout << "Sycl::finalizing" << std::endl;
    if (Sycl_initialized) {
        try {
            Sycl_initialized = false;
        }
        catch (Exception& e) {
            cerr << errstr(e) << endl;
            THROW(RuntimeError() << errmsg{"Sycl::finalize() FAILED!"});
        }
    }
    std::cout << "Sycl::finalized" << std::endl;
}
