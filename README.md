# pprank
This repository contains a parallel implementation of PageRank™, using MPI and SIMD instructions.


## Usage
On Mac OS, install the required dependencies using [Homebrew](http://brew.sh/):
```
$ brew install gcc --without-multilib
$ brew install homebrew/science/armadillo
$ brew install mpich
```

Files are formatted using [astyle](http://astyle.sourceforge.net/): `astyle --style=linux --indent-classes --indent-switches --break-closing-brackets --add-brackets --keep-one-line-statements --close-templates --max-code-length=120 --break-after-logical --lineend=linux`.


## Examples
In addition, this repository contains some minimal examples on the usage of SIMD, OpenMP and MPI:

- `examples/simd/simd.c` - Matrix-vector multiplication with SIMD instructions - `make -B simd && ./simd`

