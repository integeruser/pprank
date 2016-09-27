csrvecmul:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/csrvecmul.cpp -o csrvecmul

seqdense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/seqdense.cpp -o seqdense

seqdense-notranspose:
	g++-6 -Wall -O3 -std=c++17 -Iinclude -Ilib \
	src/ds.cpp examples/seqdense-notranspose.cpp -o seqdense-notranspose

simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd.c -o simd

clean:
	rm -f csrvecmul seqdense seqdense-notranspose simd
