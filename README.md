# pprank
This repository contains a parallel implementation of [PageRank](https://en.wikipedia.org/wiki/PageRank), using MPI and SIMD instructions.


## Usage
On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
$ brew install gcc --without-multilib
$ brew install homebrew/science/armadillo
$ HOMEBREW_CC=gcc-6 HOMEBREW_CXX=g++-6 brew install mpich --build-from-source
```

The tests included are based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests && ./tests
g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
    src/ds.cpp test/main.cpp -o tests
===============================================================================
All tests passed (29 assertions in 3 test cases)
```

The code uses [cxx-prettyprint](https://louisdx.github.io/cxx-prettyprint/) for pretty-printing C++ containers. Files are formatted with [astyle](http://astyle.sourceforge.net/) using the included options file `astylerc`.

I tested the code using two graphs from (stanford.edu)[http://snap.stanford.edu/data/#web]:


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - `make simd && ./simd` - Matrix-vector multiplication with SIMD instructions
- `examples/seqdense.cpp` - `make seqdense && ./seqdense inputs/toy.txt` - PageRank computation from a dense matrix without any parallelization
- `examples/mpi/mpi-dense-vec-product.cpp` - `make mpi-dense-vec-product && mpirun -n 2 ./mpi-dense-vec-product` - Matrix-vector multiplication with Armadillo and MPI

