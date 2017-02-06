pprank:
	mpic++ -std=c++17 -O3 -Wall -o pprank \
	src/main.cpp src/utils.cpp \
	-Iinclude -larmadillo


sequential:
	$(CXX) -std=c++17 -O3 -Wall -o sequential \
	src/utils.cpp misc/sequential.cpp \
	-Iinclude -larmadillo

notranspose:
	$(CXX) -std=c++17 -O3 -Wall -o notranspose \
	src/utils.cpp misc/notranspose.cpp \
	-Iinclude


tests:
	$(CXX) -std=c++17 -O3 -Wall -o tests \
	src/utils.cpp test/main.cpp \
	-Iinclude -larmadillo


all:
	make pprank sequential notranspose tests

clean:
	rm -f pprank sequential notranspose tests
