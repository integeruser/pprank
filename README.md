# pprank
This repository contains an [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) implementation of the [PageRank](https://en.wikipedia.org/wiki/PageRank) algorithm.


## Requisites
The code is based on [MPICH](https://www.mpich.org/) (an implementation of the MPI standard) and [Armadillo](http://arma.sourceforge.net/) (a C++ linear algebra library). It was tested with MPICH v3.2 and Armadillo v7.700.0.

On Ubuntu and similar, install these required dependencies via apt:
```
$ apt-get install build-essential mpich libarmadillo-dev
```

On Mac OS, use [Homebrew](http://brew.sh/):
```
$ brew install gcc
$ HOMEBREW_CC=gcc-6 HOMEBREW_CXX=g++-6 brew install mpich --build-from-source
$ brew install homebrew/science/armadillo
```


## Usage
Just `make pprank` and test it on the toy data set:
```
$ make pprank
mpic++ -std=c++11 -march=native -O3 -Wall -o pprank \
    src/main.cpp src/utils.cpp \
    -Iinclude -larmadillo
$ mpiexec ./pprank inputs/toy-3-2.txt
[*] Building the sparse transition matrix...[0.00 s]
        Nodes:      3
        Edges:      2
        Dangling:   1
[*] Computing PageRanks (tol=1.00e-06)...[20 iterations / 0.00 s]
        Work time:  0.00 s
        Net time:   0.00 s
[*] Writing PageRanks to file...[0.00 s]
$ cat PageRanks-3-2.txt
000000000: 1.844169e-01
000000001: 3.411710e-01
000000002: 4.744120e-01
```

`pprank` makes the following assumptions regarding the input data set:

- the data set filename must contain the number of nodes and the number of edges of the graph, matching the regular expression "(\d+)-(\d+)"
- lines starting with '#' are skipped
- zero-based node ids
- no duplicate edges
- edges ordered by source node id
- the data set file ends with a newline


## Misc
I also included some tests based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests
g++-6 -std=c++11 -march=native -O3 -Wall -o tests \
    src/utils.cpp test/main.cpp \
    -Iinclude -larmadillo
$ ./tests
===============================================================================
All tests passed (39 assertions in 3 test cases)
```
