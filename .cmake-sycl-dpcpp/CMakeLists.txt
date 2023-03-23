cmake_minimum_required(VERSION 3.1.14 FATAL_ERROR)
project(wire-cell-gen-sycl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)


file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

find_package(Boost REQUIRED COMPONENTS)
set(Boost_USE_MULTITHREADED ON)

find_package(spdlog REQUIRED )
find_package (Eigen3  REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)

find_package(jsoncpp REQUIRED)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)


#string(APPEND CMAKE_CXX_FLAGS "-g -O3 -fopenmp -pedantic -Wall -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906")
string(APPEND CMAKE_CXX_FLAGS " -g -O3 -fopenmp -pedantic -Wall -fsycl -fsycl-targets=spir64_x86_64 ")
#string(APPEND CMAKE_CXX_FLAGS " -g -O3 -fopenmp -pedantic -Wall -fsycl ")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenSycl SHARED ${all_files})


target_include_directories(WireCellGenSycl
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{RANDOM123_INC}
    ${JSON_INC_PATH}
    $ENV{JSONNET_INC}
    $ENV{WIRECELL_INC}
)
#set_target_properties(WireCellGenSycl
#	PROPERTIES COMPILE_OPTIONS "-D__SYCL_DISABLE_NAMESPACE_INLINE__;-DEIGEN_NO_CUDA")
target_link_directories(WireCellGenSycl PRIVATE  $ENV{WIRECELL_LIB}  )
target_link_libraries(WireCellGenSycl PRIVATE  WireCellIface WireCellUtil WireCellAux Eigen3::Eigen jsoncpp_lib PkgConfig::FFTW spdlog::spdlog Boost::headers)

add_subdirectory(test)
