# pprank
This repository contains a parallel implementation of [PageRank](https://en.wikipedia.org/wiki/PageRank), using MPI and SIMD instructions.


## Requisites
The code is based on [Armadillo](http://arma.sourceforge.net/) `v7.600.2` (a C++ linear algebra library) and [MPICH](https://www.mpich.org/) `v3.2` (an implementation of the MPI standard). The code was compiled using [GCC](https://gcc.gnu.org/) `v6.3.0`.

On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
$ brew install gcc --without-multilib
$ brew install homebrew/science/armadillo
$ HOMEBREW_CC=gcc-6 HOMEBREW_CXX=g++-6 brew install mpich --build-from-source
```


## Usage
The tests included are based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests && ./tests
g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
    src/ds.cpp test/main.cpp -o tests
===============================================================================
All tests passed (29 assertions in 3 test cases)
```

The code uses [cxx-prettyprint](https://louisdx.github.io/cxx-prettyprint/) for pretty-printing C++ containers. Files are formatted with [astyle](http://astyle.sourceforge.net/) using the included options file `astylerc`.

I tested the code using two graphs from [stanford.edu](http://snap.stanford.edu/data/#web):


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - `make simd && ./simd` - Matrix-vector multiplication with SIMD instructions
- `examples/seqdense.cpp` - `make seqdense && ./seqdense inputs/toy.txt` - PageRank computation from a dense matrix without any parallelization
- `examples/mpi-matvecprod-dense.cpp` - `make mpi-matvecprod-dense && mpirun -n 2 ./mpi-matvecprod-dense` - Matrix-vector multiplication with Armadillo and MPI

