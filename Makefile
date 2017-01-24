pprank:
	mpic++ -std=c++17 -O3 -Wall -o pprank \
	src/main.cpp src/utils.cpp \
	-Iinclude -larmadillo


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


tests:
	$(CXX) -std=c++17 -O3 -Wall -o tests \
	src/utils.cpp test/main.cpp \
	-Iinclude -larmadillo


all:
	make pprank pagerank-notranspose pagerank pagerank-csr tests

clean:
	rm -f pprank pagerank-notranspose pagerank pagerank-csr tests
