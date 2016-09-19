simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd.c -o simd

clean:
	rm -f simd
