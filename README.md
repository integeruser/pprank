# pprank
This repository contains a parallel implementation of [PageRank](https://en.wikipedia.org/wiki/PageRank), using MPI and SIMD instructions.


## Usage
On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
$ brew install gcc --without-multilib
$ brew install homebrew/science/armadillo
$ brew install mpich
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


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - `make simd && ./simd`  
Matrix-vector multiplication with SIMD instructions
- `examples/simd/seqdense.cpp` - `make seqdense && ./seqdense inputs/toy.txt`  
PageRank computation from a dense matrix without any parallelization

