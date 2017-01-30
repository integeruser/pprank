# pprank
This repository contains an [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) implementation of distributed [PageRank](https://en.wikipedia.org/wiki/PageRank).


## Requisites
The code is based on [MPICH](https://www.mpich.org/) v3.2 (an implementation of the MPI standard) and [Armadillo](http://arma.sourceforge.net/) v7.600.2 (a C++ linear algebra library).

On Ubuntu and similar, install the required dependencies via apt:
```
$ apt-get install build-essential mpich libarmadillo-dev
```

On Mac OS, use [Homebrew](http://brew.sh/):
```
$ brew install gcc
$ HOMEBREW_CC=gcc-6 HOMEBREW_CXX=g++-6 brew install mpich --build-from-source
$ brew install homebrew/science/armadillo
```

The code uses [cxx-prettyprint](https://louisdx.github.io/cxx-prettyprint/) for pretty-printing C++ containers. Files are formatted with [astyle](http://astyle.sourceforge.net/) using the included options file `astylerc`.


## Usage
Just compile `pprank` and run it on the toy data set:
```
$ make pprank && mpiexec ./pprank inputs/toy.txt
mpic++ -std=c++17 -O3 -Wall -o pprank \
    src/main.cpp src/utils.cpp \
    -Iinclude -larmadillo
[*] Building graph...
        Nodes: 3
[*] Building adjacency matrix...
[*] Finding dangling nodes...
[*] Splitting matrix in blocks...
[*] Computing PageRank (tol=1e-06)...
[*] Ranks: [(0, 0.184417), (1, 0.341171), (2, 0.474413)] in 20 iterations
```

The tests included are based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests && ./tests
g++-6 -std=c++17 -O3 -Wall -o tests \
    src/utils.cpp test/main.cpp \
    -Iinclude -larmadillo
===============================================================================
All tests passed (19 assertions in 3 test cases)
```


## Results
I tested the code using two graphs from [stanford.edu](http://snap.stanford.edu/data/#web):
TODO


## Misc
In addition to the MPI implementation, the `misc` folder contains:

- `examples/pagerank-dense.cpp` - `make pagerank-dense && ./pagerank-dense inputs/toy.txt` - PageRank computation from a dense matrix
- `examples/pagerank-notranspose.cpp` - `make pagerank-notranspose && ./pagerank-notranspose inputs/toy.txt` - PageRank computation without storing the ranks matrix in memory (but recomputing its rows at every iteration)
- `examples/mpi-matvecprod-dense.cpp` - `make mpi-matvecprod-dense && mpirun -n 2 ./mpi-matvecprod-dense` - Matrix-vector multiplication with Armadillo and MPI
- `examples/simd/simd.c` - `make simd && ./simd` - Matrix-vector multiplication with SIMD instructions
