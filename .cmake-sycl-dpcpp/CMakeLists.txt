cmake_minimum_required(VERSION 3.1.14 FATAL_ERROR)
project(wire-cell-gen-sycl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)


file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)



set(BOOST_ROOT $ENV{BOOST_DIR} )
#set(BOOST_INCLUDEDIR $ENV{BOOST_INC})
#set(BOOST_LIBRARYDIR $ENV{BOOST_LIB})

find_package(Boost REQUIRED COMPONENTS)
#include_directories(Boost_INCLUDE_DIRS})
set(Boost_USE_MULTITHREADED ON)


#string(APPEND CMAKE_CXX_FLAGS "-g -O3 -fopenmp -pedantic -Wall -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906")
string(APPEND CMAKE_CXX_FLAGS "-g -O3 -fopenmp -pedantic -Wall -fsycl -fsycl-targets=spir64_x86_64 -Xs \"-march=avx2\"   ")
#string(APPEND CMAKE_CXX_FLAGS "-g -O3 -fopenmp -pedantic -Wall -fsycl ")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenSycl SHARED ${all_files})


target_include_directories(WireCellGenSycl
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{RANDOM123_INC}
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenSycl
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE")
target_link_directories(WireCellGenSycl PRIVATE  $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB}  )
target_link_libraries(WireCellGenSycl PRIVATE  jsoncpp WireCellIface WireCellUtil Boost::headers)

add_subdirectory(test)
