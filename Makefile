csrsplit:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/csrsplit.cpp -o csrsplit

seqdense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/seqdense.cpp -o seqdense

seqdense-notranspose:
	g++-6 -Wall -O3 -std=c++17 -Iinclude -Ilib \
	src/ds.cpp examples/seqdense-notranspose.cpp -o seqdense-notranspose

mpi-dense-vec-product:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -lmpi -Iinclude \
	src/nw.cpp examples/mpi-dense-vec-product.cpp -o mpi-dense-vec-product

simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd/simd.c -o simd


tests:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp test/main.cpp -o tests


clean:
	rm -f mpi-dense-vec-product seqdense seqdense-notranspose simd tests
