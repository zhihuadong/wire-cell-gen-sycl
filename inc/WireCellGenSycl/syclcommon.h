#ifndef GENSYCL_SYCLCOMMON_H_
#define GENSYCL_SYCLCOMMON_H_
#include <CL/sycl.hpp>

//#define SYCL_TARGET_CUDA 

namespace WireCell::GenSycl::syclcommon {

#ifdef SYCL_TARGET_CUDA
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();

    if (device.is_gpu() &&
        (device_vendor.find("NVIDIA") != std::string::npos) &&
        (device_driver.find("CUDA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};
#endif
#ifdef SYCL_TARGET_HIP
class AMDSelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name = device.get_info<cl::sycl::info::device::name>();

    if (device.is_gpu() && (device_vendor.find("AMD") != std::string::npos)) {
      return 1;
    }
    return -1;
  }
};
#endif

// Gets the target device, as defined in the cmake configuration.
static inline cl::sycl::device GetTargetDevice() {
  cl::sycl::device dev;
#if defined SYCL_TARGET_CUDA
#warning sycl_cuda
  CUDASelector cuda_selector;
  try {
    dev = cl::sycl::device(cuda_selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_HIP
  AMDSelector selector;
  try {
    dev = cl::sycl::device(selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_DEFAULT
  dev = cl::sycl::device(cl::sycl::default_selector());
#elif defined SYCL_TARGET_CPU
  dev = cl::sycl::device(cl::sycl::cpu_selector());
#elif defined SYCL_TARGET_GPU
  dev = cl::sycl::device(cl::sycl::gpu_selector());
#else
  dev = cl::sycl::device(cl::sycl::host_selector());
#endif

  return dev;
}

static inline cl::sycl::context GetSharedContext() {
  cl::sycl::platform platform;
#if defined SYCL_TARGET_CUDA
  CUDASelector cuda_selector;
  try {
    platform = cl::sycl::platform(cuda_selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_DEFAULT
  platform = cl::sycl::platform(cl::sycl::default_selector());
#elif defined SYCL_TARGET_CPU
  platform = cl::sycl::platform(cl::sycl::cpu_selector());
#elif defined SYCL_TARGET_GPU
  platform = cl::sycl::platform(cl::sycl::gpu_selector());
#else
  platform = cl::sycl::platform(cl::sycl::host_selector());

#endif

  return cl::sycl::context(platform);
}

}  // namespace GenSycl::syclcommon

#endif  // GENSYCL_SYCLCOMMON_H_
