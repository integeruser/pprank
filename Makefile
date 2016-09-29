csrsplit:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/csrsplit.cpp -o csrsplit

seqdense:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp examples/seqdense.cpp -o seqdense

seqdense-notranspose:
	g++-6 -Wall -O3 -std=c++17 -Iinclude -Ilib \
	src/ds.cpp examples/seqdense-notranspose.cpp -o seqdense-notranspose

simd:
	gcc-6 -Wall -O3 -std=c11 -msse4.1 examples/simd.c -o simd


tests:
	g++-6 -Wall -O3 -std=c++17 -larmadillo -Iinclude -Ilib \
	src/ds.cpp test/main.cpp -o tests


clean:
	rm -f csrsplit seqdense seqdense-notranspose simd
