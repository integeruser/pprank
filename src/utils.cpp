#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "utils.hpp"

#include "armadillo"


Graph::Graph(const std::string& filename)
{
    // assume the file contains two integers per line, separated by a whitespace
    // each line represent an edge from a source node to a destination node
    std::ifstream file(filename, std::ios::in);

    num_nodes = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        uint_fast32_t from_node, to_node;
        if (!(iss >> from_node >> to_node)) {
            // skip malformed lines
            continue;
        }

        edges[from_node].insert(to_node);
        inv_edges[to_node].insert(from_node);

        num_nodes = std::max(std::max(from_node, to_node), num_nodes);
    }
    // assume the node ids start from zero
    ++num_nodes;

    file.close();
}


arma::sp_fmat to_adjacency_mat(const Graph& graph)
{
    // convert the graph to a sparse adjacency matrix
    arma::sp_fmat adjacency_mat(graph.num_nodes, graph.num_nodes);
    for (uint_fast32_t i = 0; i < graph.num_nodes; ++i) {
        const auto outdegree = graph.edges.count(i) > 0 ? graph.edges.at(i).size() : 0;
        if (outdegree == 0) {
            // dangling node
            for (uint_fast32_t j = 0; j < graph.num_nodes; ++j) {
                adjacency_mat(i, j) = 1.0f/graph.num_nodes;
            }
        }
        else {
            for (uint_fast32_t j: graph.edges.at(i)) {
                adjacency_mat(i, j) = 1.0f/outdegree;
            }
        }
    }
    return adjacency_mat;
}


CSC::CSC(const Graph& graph)
{
    num_rows = num_cols = graph.num_nodes;

    // find nodes without outgoing edges
    std::set<uint_fast32_t> dangling_nodes;
    for (uint_fast32_t node = 0; node < graph.num_nodes; ++node) {
        if (graph.edges.count(node) == 0) {
            dangling_nodes.insert(node);
        }
    }

    uint_fast32_t num_values = 0;
    ia.push_back(num_values);

    for (uint_fast32_t to_node = 0; to_node < graph.num_nodes; ++to_node) {
        const auto& in_nodes = graph.inv_edges.count(to_node) > 0 ?
                               graph.inv_edges.at(to_node) :
                               std::set<uint_fast32_t>();

        // find which rows have a value (different from zero)
        auto from_nodes = in_nodes;
        from_nodes.insert(dangling_nodes.begin(), dangling_nodes.end());

        // store the current column in csc
        for (auto from_node: from_nodes) {
            const float weight = dangling_nodes.count(from_node) > 0 ?
                                 1.0f/graph.num_nodes :
                                 1.0f/in_nodes.size();
            a.push_back(weight);
            ja.push_back(from_node);
        }
        num_values += from_nodes.size();
        ia.push_back(num_values);
    }
}


arma::fvec CSR::operator*(const arma::fvec& vec) const
{
    // see http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html
    arma::fvec x = arma::zeros<arma::fvec>(num_rows);
    for (std::size_t i = 0; i < num_rows; ++i) {
        if (ia[i] == ia[i+1]) {
            // TO CHECK
            // std::cout << "SUCA" << std::endl;
            // for (std::size_t k = 0; k < num_rows; ++k) {
            //     x[i] += (1.0f/num_rows) * vec[k];
            // }
        }
        else {
            for (std::size_t k = ia[i]; k < ia[i+1]; ++k) {
                x[i] = x[i] + a[k] * vec[ja[k]];
            }
        }
    }
    return x;
}


CSR transpose(const CSC& csc)
{
    CSR csr;
    csr.num_rows = csc.num_cols;
    csr.num_cols = csc.num_rows;
    csr.a = csc.a;
    csr.ia = csc.ia;
    csr.ja = csc.ja;
    return csr;
}
