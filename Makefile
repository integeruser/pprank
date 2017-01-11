pagerank-dense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/pagerank-dense.cpp -o pagerank-dense

pagerank-notranspose:
	g++-6 -Wall -O3 -std=c++17 -Iinclude -Ilib \
	src/ds.cpp examples/pagerank-notranspose.cpp -o pagerank-notranspose


mpi-matvecprod-dense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/nw.cpp examples/mpi-matvecprod-dense.cpp -o mpi-matvecprod-dense

simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd/simd.c -o simd


tests:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp test/main.cpp -o tests


clean:
	rm -f pagerank-dense pagerank-notranspose mpi-matvecprod-dense simd tests
