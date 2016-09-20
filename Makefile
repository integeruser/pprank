seqdense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/seqdense.cpp -o seqdense

simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd.c -o simd

clean:
	rm -f seqdense simd
