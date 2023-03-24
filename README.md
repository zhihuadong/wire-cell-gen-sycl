# SYCL version of the wire-cell-gen Standardalone verison



## prerequisites
 - Need to build Wire-Cell-toolkit   and its dependencies.
   Best use spack.
```bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
which spack
git clone https://github.com/WireCell/wire-cell-spack.git wct-spack
spack repo add wct-spack
#use your own os, here we use ubuntu20.04 as example
#by default spack will detect your OS and CPU 
#But to avoid a issue of inconsistant arch from different compilers we choose x86_64   
spack install wire-cell-toolkit arch=linux-ubuntu20.04-x86_64

```
   

     
 - Need to have sycl compiler installed :
   Suggest install oneAPI basekit for cpu backend .
   for Nvidia and AMD GPU
   Please download and install plugins from 
   https://codeplay.com/solutions/oneapi

 - CUDA and ROCM (include hipfft) need to be installed on system for GPU backend

### Download code
```bash
git clone https://github.com/WireCell/wire-cell-gen-sycl.git 
git checkout wc20
export WC_BUILD_DIR=/your/build/directory
export WC_SYCL_SRC_DIR=${PWD}/wire-cell-gen-sycl
#dependency code RNG wrapper. 
git clone https://github.com/DEShawResearch/random123.git
git clone https://github.com/GKNB/test-benchmark-OpenMP-RNG.git omprng
export RANDOM123_INC=${PWD}/random123/include
export OMGRNG=${PWD}/omprng

```

### build 

### setup enviroments:
```bash
spack load wire-cell-toolkit
export WIRECELL_DIR=$(spack find -p wire-cell-toolkit |grep wire |awk '{print $2}'
export WIRECELL_INC=${WIRECELL_DIR}/include
export WIRECELL_LIB=${WIRECELL_DIR}/lib
export JSONNET_DIR=$(spack find -p go-jsonnet |grep go-jsonnet|awk '{print $2}'
export JSONNET_INC=${JSONNET_DIR}/include
```


### build host (cpu backend)
```bash
cmake -B ${WC_BUILD_DIR} $SC_SYCL_SRC_DIR/.cmake-sycl-dpcpp
make -C ${WC_BUILD_DIR} -j 10
```
### build for Nvidia GPU backend
```bash
cmake -B ${WC_BUILD_DIR} $SC_SYCL_SRC_DIR/.cmake-sycl
make -C ${WC_BUILD_DIR} -j 10
```

### build for AMD GPU backend
```bash
export HIP_DIR=/opt/rocm/hip
#specify local amd GPU arch
export SYCL_CLANG_EXTRA_FLAGS=" -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 "
cmake -B ${WC_BUILD_DIR} $SC_SYCL_SRC_DIR/.cmake-sycl-amd
make -C ${WC_BUILD_DIR} -j 10
```
## test running 


### download `wire-cell-data`

`wire-cell-data` contains needed data files, e.g. geometry files, running full tests.

```
git clone https://github.com/WireCell/wire-cell-data.git
```

### downlaod `wire-cell-toolkit`
We need the config files there.
```
 git clone https://github.com/WireCell/wire-cell-toolkit.git
```

 

### setup `$WIRECELL_PATH`

`wire-cell` searches pathes in this env var for configuration and data files.

for bash, run something like this below:

```
export WIRECELL_PATH=$WCT_SRC/cfg #  main CFG
export WIRECELL_PATH=$WIRECELL_DATA_PATH:$WIRECELL_PATH # data
export WIRECELL_PATH=$WC_SYCL_SRC/cfg:$WIRECELL_PATH # gen-sycl


```
Variable meaning:
 - `$WCT_SRC` is a `wire-cell-toolkit` source file directory
 - `WIRECELL_DATA_PATH` refer to the git repository cloned from the previous step
 - `WC_SYCL_SRC` refer to the  path of the `wire-cell-gen-sycl` standalone SRC.

### $LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=${WC_BUILD_DIR}:$LD_LIBRARY_PATH
```

### Download sample input file:
e.g.   
```
https://www.phy.bnl.gov/~yuhw/kokkos/sample/wire-cell-sio/pdsp/cosmic-500/1/depos.tar.bz2
```

### run
Use the `wct-sim.jsonnet` in  sample directory 
```
wire-cell -l stdout -L debug \
-V input="depos.tar.bz2" \
-V output="frames.tar.bz2" \
-c wct-sim.jsonnet
```

