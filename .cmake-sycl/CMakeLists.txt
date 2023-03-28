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

find_package(CUDA REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -g -O3  -pedantic -Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda ")
#string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenSycl SHARED ${all_files})
target_include_directories(WireCellGenSycl
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    ${JSON_INC_PATH}
    $ENV{JSONNET_INC}
    $ENV{WIRECELL_INC}
)
#set_target_properties(WireCellGenSycl
#	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE;-DSYCL_TARGET_CUDA")
set_target_properties(WireCellGenSycl
	PROPERTIES COMPILE_OPTIONS "-DSYCL_TARGET_CUDA")

target_link_directories(WireCellGenSycl PRIVATE $ENV{WIRECELL_LIB} )
target_link_libraries(WireCellGenSycl PRIVATE  jsoncpp_lib WireCellIface WireCellUtil Eigen3::Eigen  WireCellAux spdlog::spdlog Boost::headers ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} )

add_subdirectory(test)
