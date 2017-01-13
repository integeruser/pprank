pagerank:
	g++-6 -std=c++17 -O3 -Wall -o pagerank \
	src/ds.cpp examples/pagerank.cpp \
	-Iinclude -Ilib -larmadillo

pagerank-notranspose:
	g++-6 -Wall -O3 -std=c++17 -Iinclude -Ilib \
	src/ds.cpp examples/pagerank-notranspose.cpp -o pagerank-notranspose


mpi-matvecprod-dense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/nw.cpp examples/mpi-matvecprod-dense.cpp -o mpi-matvecprod-dense

mpi-pagerank-dense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude -Ilib \
	src/ds.cpp src/nw.cpp examples/mpi-pagerank-dense.cpp -o mpi-pagerank-dense
simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd/simd.c -o simd


tests:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp test/main.cpp -o tests


clean:
	rm -f pagerank pagerank-notranspose mpi-matvecprod-dense mpi-pagerank-dense simd tests
