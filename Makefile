pagerank-notranspose:
	$(CXX) -std=c++17 -O3 -Wall -o pagerank-notranspose \
	src/utils.cpp examples/pagerank-notranspose.cpp \
	-Iinclude

pagerank:
	$(CXX) -std=c++17 -O3 -Wall -o pagerank \
	src/utils.cpp examples/pagerank.cpp \
	-Iinclude -larmadillo

pagerank-csr:
	$(CXX) -std=c++17 -O3 -Wall -o pagerank-csr \
	src/utils.cpp examples/pagerank-csr.cpp \
	-Iinclude -larmadillo


mpi-pagerank-csr:
	$(CXX) -std=c++17 -O3 -Wall -o mpi-pagerank-csr \
	src/nw.cpp src/utils.cpp examples/mpi-pagerank-csr.cpp \
	-Iinclude -larmadillo -lmpi


all:
	make pagerank-notranspose pagerank pagerank-csr mpi-pagerank-csr

clean:
	rm -f pagerank-notranspose pagerank pagerank-csr mpi-pagerank-csr
