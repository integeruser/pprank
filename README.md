# pprank
This repository contains a parallel implementation of [PageRank](https://en.wikipedia.org/wiki/PageRank), using MPI and SIMD instructions.


## Usage
On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
$ brew install gcc --without-multilib
$ brew install homebrew/science/armadillo
$ brew install mpich
```

Files are formatted using [astyle](http://astyle.sourceforge.net/) with the command `astyle --options=astylerc ./file/to/format`.


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - `make -B simd && ./simd`  
Matrix-vector multiplication with SIMD instructions
- `examples/simd/seqdense.cpp` - `make -B seqdense && ./seqdense inputs/toy.txt`  
PageRank computation from a dense matrix without any parallelization

