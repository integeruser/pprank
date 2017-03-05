# pprank
This repository contains a simple implementation of the [PageRank](https://en.wikipedia.org/wiki/PageRank) algorithm based on the [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) standard. Since PageRank does not scale well on multiple computational nodes for the high volume of data that needs to be exchanged, this project is mostly an academic exercise.


## Requisites
The code depends on [Armadillo](http://arma.sourceforge.net/) (a linear algebra library) and was tested using [MPICH](https://www.mpich.org/) v3.2.

On Ubuntu and similar, install these dependencies via apt:
```
$ apt-get install build-essential mpich libarmadillo-dev
```

On Mac OS, use [Homebrew](http://brew.sh/) (`gcc-6` not strictly required):
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
    src/pprank.cpp src/utils.cpp \
    -Iinclude -larmadillo
$ mpiexec -n 2 ./pprank inputs/toy-3-2.txt
[*] Building the sparse transition matrix...[0.00 s]
        Nodes:      3
        Edges:      2
        Dangling:   1
[*] Computing PageRanks (tol=1.00e-06)...[20 iterations / 0.00 s]
        (MASTER) Work time: 0.00 s
        (MASTER) Netw time: 0.00 s
[*] Writing PageRanks to file...[0.00 s]
$ cat PageRanks-3-2.txt
000000000: 1.844169e-01
000000001: 3.411710e-01
000000002: 4.744120e-01
```

As specified in `src/utils.cpp`, the following assumptions are made for the input data set:

- the filename must contain the number of nodes and the number of edges of the graph, matching the regular expression "(\d+)-(\d+)"
- lines starting with '#' are not parsed
- node ids are zero-based
- there are no duplicate edges
- edges are ordered by source node id
- it must end with a newline


## Misc
In this repository, I also included a sequential version of PageRank (see `src/sequential.cpp`) and a few tests based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests
g++-6 -std=c++11 -march=native -O3 -Wall -o tests \
	src/utils.cpp src/tests.cpp \
	-Iinclude -larmadillo
~/G/s/u/m/I/h/pprank (master|✔) $ ./tests
===============================================================================
All tests passed (45 assertions in 3 test cases)
```
