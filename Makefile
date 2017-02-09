pprank:
	mpic++ -std=c++17 -O3 -Wall -o pprank \
	src/main.cpp src/utils.cpp \
	-Iinclude -larmadillo


sequential:
	$(CXX) -std=c++17 -O3 -Wall -o sequential \
	src/utils.cpp misc/sequential.cpp \
	-Iinclude -larmadillo


tests:
	$(CXX) -std=c++17 -O3 -Wall -o tests \
	src/utils.cpp test/main.cpp \
	-Iinclude -larmadillo


all:
	make pprank sequential tests

clean:
	rm -f pprank sequential tests
