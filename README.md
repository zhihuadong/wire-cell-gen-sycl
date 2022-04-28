# SYCL version of the wire-cell-gen

## prerequisites
 - Need to have access to a `Wire-Cell Toolkit` build and its dependencies.
 - Need to have access to a `SYCL` build.
 - Need to have sycl compiler installed  https://codeplay.com/solutions/oneapi/for-cuda/#getting-started
 - Need to have oneMKL Open Source Interface Library installed https://github.com/oneapi-src/oneMKL 



```bash
git clone https://github.com/WireCell/wire-cell-gen-sycl.git 
export WC_BUILD_DIR=/your/build/directory
export WC_SYCL_SRC_DIR=${PWD}/wire-cell-gen-sycl
cmake -B ${WC_BUILD_DIR} $SC_SYCL_SRC_DIR/.cmake-sycl
make -C ${WC_BUILD_DIR} -j 10
```

## full test running `wire-cell` as plugin of `LArSoft`


### download `wire-cell-data`

`wire-cell-data` contains needed data files, e.g. geometry files, running full tests.

```
git clone https://github.com/WireCell/wire-cell-data.git
```

### $WIRECELL_PATH

`wire-cell` searches pathes in this env var for configuration and data files.

for bash, run something like this below:

```
export WIRECELL_PATH=$WIRECELL_FQ_DIR/wirecell-0.14.0/cfg # main cfg
export WIRECELL_PATH=$WIRECELL_DATA_PATH:$WIRECELL_PATH # data
export WIRECELL_PATH=$WC_SYCL_SRC/cfg:$WIRECELL_PATH # gen-sycl


```
Variable meaning:
 - `$WIRECELL_FQ_DIR` is a variable defined developing in Kyle's container or `setup wirecell` in a Fermilab ups system, current version is `0.14.0`, may upgrade in the future.
 - `WIRECELL_DATA_PATH` refer to the git repository cloned from the previous step
 - `WC_SYCL_SRC` refer to the  path of the `wire-cell-gen-sycl` standalone SRC.

### $LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=${WC_BUILD_DIR}:$LD_LIBRARY_PATH
```


### run

 - input: a root file (refered to as [`g4.root`](https://github.com/hep-cce2/PPSwork/blob/master/Wire-Cell/examples/g4.root) below) containing Geant4 energy depo (`sim::SimEnergyDeposits`)
 - in the example folder: `lar -n 1 -c sim.fcl g4.root`

