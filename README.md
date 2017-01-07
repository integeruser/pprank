# pprank
This repository contains a parallel implementation of PageRank™, using MPI and SIMD instructions.


## Usage
On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
brew install gcc --without-multilib homebrew/science/armadillo mpich
```


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - Matrix-vector multiplication with SIMD instructions - `make -B simd && ./simd`

