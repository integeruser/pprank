pprank:
	mpic++ -std=c++11 -march=native -O3 -Wall -o pprank \
	src/pprank.cpp src/utils.cpp \
	-Iinclude -larmadillo

sequential:
	$(CXX) -std=c++11 -march=native -O3 -Wall -o sequential \
	src/sequential.cpp src/utils.cpp \
	-Iinclude -larmadillo

tests:
	$(CXX) -std=c++11 -march=native -O3 -Wall -o tests \
	src/utils.cpp src/tests.cpp \
	-Iinclude -larmadillo


all:
	make pprank sequential tests

clean:
	rm -f pprank sequential tests
