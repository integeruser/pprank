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


mpi-matvecprod-dense:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/nw.cpp examples/mpi-matvecprod-dense.cpp -o mpi-matvecprod-dense

mpi-pagerank-dense:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/ds.cpp src/nw.cpp examples/mpi-pagerank-dense.cpp -o mpi-pagerank-dense
ompcpp:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -fopenmp -Iinclude \
	examples/omp.cpp -o ompcpp


simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd/simd.c -o simd



pprank:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/ds.cpp src/nw.cpp src/main.cpp -o pprank


tests:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -Iinclude \
	src/ds.cpp test/main.cpp -o tests


clean:
	rm -f pagerank pagerank-notranspose mpi-matvecprod-dense mpi-pagerank-dense simd tests



prove:
	$(CXX) -Wall -O3 -std=c++17 -larmadillo -Iinclude \
	prove.cpp -o prove
