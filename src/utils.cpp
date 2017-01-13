#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

#include "utils.hpp"

#include "armadillo"


arma::sp_fmat load(const std::string& filename)
{
    std::map<uint_fast32_t, std::set<uint_fast32_t>> edges;

    // assume the file contains two integers per line, separated by a whitespace
    // each line represent an edge from a source node to a destination node
    std::ifstream file(filename, std::ios::in);

    uint_fast32_t num_nodes = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        uint_fast32_t from_node, to_node;
        if (!(iss >> from_node >> to_node)) {
            // skip malformed lines
            continue;
        }

        edges[from_node].insert(to_node);
        num_nodes = 1 + std::max(std::max(from_node, to_node), num_nodes);
    }

    file.close();

    // convert the graph to a sparse adjacency matrix
    arma::sp_fmat adjacency_mat(num_nodes, num_nodes);
    for (uint_fast32_t i = 0; i < num_nodes; ++i) {
        const auto outdegree = edges.count(i) > 0 ? edges.at(i).size() : 0;
        if (outdegree == 0) {
            // dangling node
            for (uint_fast32_t j = 0; j < num_nodes; ++j) {
                adjacency_mat(i, j) = 1.0f/num_nodes;
            }
        }
        else {
            for (uint_fast32_t j: edges.at(i)) {
                adjacency_mat(i, j) = 1.0f/outdegree;
            }
        }
    }
    return adjacency_mat;
}
