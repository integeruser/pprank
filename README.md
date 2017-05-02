# pprank
This repository contains a simple implementation of the [PageRank](https://en.wikipedia.org/wiki/PageRank) algorithm based on the [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) standard. Since PageRank does not scale well on multiple computational nodes for the high volume of data exchanged, this project is mostly an academic exercise.


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


## Tests
Results obtained on the [LiveJournal social network data set](http://snap.stanford.edu/data/soc-LiveJournal1.html) using a [MacBook Pro](https://support.apple.com/kb/SP690) with a 2 GHz quad-core Intel Core i7 processor and 8 GB of 1600 MHz DDR3L RAM:
```
$ mpiexec -n 1 ./pprank ./soc-LiveJournal1-4847571-68993773.txt
[*] Building the sparse transition matrix...[7.06 s]
        Nodes:      4847571
        Edges:      68993773
        Dangling:   539119
[*] Computing PageRanks (tol=1.00e-06)...[51 iterations / 38.78 s]
        (MASTER) Work time: 33.54 s
        (MASTER) Netw time: 0.45 s
[*] Writing PageRanks to file...[9.13 s]

$ mpiexec -n 2 ./pprank ./soc-LiveJournal1-4847571-68993773.txt
[*] Building the sparse transition matrix...[6.02 s]
        Nodes:      4847571
        Edges:      68993773
        Dangling:   539119
[*] Computing PageRanks (tol=1.00e-06)...[51 iterations / 36.74 s]
        (MASTER) Work time: 29.01 s
        (MASTER) Netw time: 1.10 s
[*] Writing PageRanks to file...[9.54 s]

$ mpiexec -n 3 ./pprank ./soc-LiveJournal1-4847571-68993773.txt
[*] Building the sparse transition matrix...[8.83 s]
        Nodes:      4847571
        Edges:      68993773
        Dangling:   539119
[*] Computing PageRanks (tol=1.00e-06)...[51 iterations / 36.72 s]
        (MASTER) Work time: 26.74 s
        (MASTER) Netw time: 1.52 s
[*] Writing PageRanks to file...[10.06 s]

$ mpiexec -n 4 ./pprank ./soc-LiveJournal1-4847571-68993773.txt
[*] Building the sparse transition matrix...[7.89 s]
        Nodes:      4847571
        Edges:      68993773
        Dangling:   539119
[*] Computing PageRanks (tol=1.00e-06)...[51 iterations / 35.29 s]
        (MASTER) Work time: 23.68 s
        (MASTER) Netw time: 1.43 s
[*] Writing PageRanks to file...[10.65 s]
```


## Misc
In this repository, I also included a sequential version of PageRank which does not require an MPI implementation (see `src/sequential.cpp`) and a few tests based on the [Catch](https://github.com/philsquared/Catch) framework:
```
$ make tests
g++-6 -std=c++11 -march=native -O3 -Wall -o tests \
	src/utils.cpp src/tests.cpp \
	-Iinclude -larmadillo
$ ./tests
===============================================================================
All tests passed (45 assertions in 3 test cases)
```
