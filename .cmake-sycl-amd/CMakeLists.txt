cmake_minimum_required(VERSION 3.1.14 FATAL_ERROR)
project(wire-cell-gen-sycl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)


file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)


find_package(Boost REQUIRED COMPONENTS)
set(Boost_USE_MULTITHREADED ON)

find_package(spdlog REQUIRED )
find_package (Eigen3  REQUIRED NO_MODULE)

#find_package(PkgConfig REQUIRED)
#pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)

find_package(jsoncpp REQUIRED)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)


#string(APPEND CMAKE_CXX_FLAGS "-g -O3 -pedantic -Wall -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=$ENV{AMD_ARCH}")
#string(APPEND CMAKE_CXX_FLAGS "-g -O3 -fopenmp -pedantic -Wall -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906")
string(APPEND CMAKE_CXX_FLAGS "-g -O3  -pedantic -Wall -fsycl $ENV{SYCL_CLANG_EXTRA_FLAGS}")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenSycl SHARED ${all_files})

list(APPEND CMAKE_PREFIX_PATH /opt/rocm )

find_package(rocrand) 
find_package(hipfft)

target_include_directories(WireCellGenSycl
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{HIP_DIR}/include
    ${JSON_INC_PATH}
    $ENV{JSONNET_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenSycl
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE;-DSYCL_TARGET_HIP;-D__HIP_PLATFORM_AMD__")
target_link_directories(WireCellGenSycl PRIVATE  $ENV{WIRECELL_LIB} $ENV{HIP_DIR}/lib  } )
target_link_directories(WireCellGenSycl PRIVATE  $ENV{WIRECELL_LIB} )
target_link_libraries(WireCellGenSycl PRIVATE  jsoncpp_lib WireCellIface WireCellUtil WireCellAux  Eigen3::Eigen spdlog::spdlog Boost::headers roc::rocrand hip::hipfft amdhip64 dl)

add_subdirectory(test)
