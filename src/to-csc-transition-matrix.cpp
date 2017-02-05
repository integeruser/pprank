#include <cstdint>
#include <fstream>
#include <iostream>

#include "utils.hpp"


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: to-csc-transition-matrix file" << std::endl;
        return EXIT_FAILURE;
    }

    const char* file = argv[1];

    std::cout << "[*] Building graph..." << std::endl;
    const Graph graph = Graph(file);
    std::cout << "        Nodes: " << graph.num_nodes << std::endl;

    std::cout << "[*] Building CSC transition matrix..." << std::endl;
    const CSC csc = CSC(graph);
    std::cout << "        Rows:      " << csc.num_rows << std::endl;
    std::cout << "        Columns:   " << csc.num_cols << std::endl;
    std::cout << "        Nonzeros:  " << csc.a.size() << std::endl;
    std::cout << "        Danglings: " << csc.dangling_nodes.size() << std::endl;

    std::cout << "[*] Writing CSC transition matrix to file..." << std::endl;
    csc.to_file();

    return EXIT_SUCCESS;
}
