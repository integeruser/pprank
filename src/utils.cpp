#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

        in_edges[to_node].insert(from_node);
        out_edges[from_node].insert(to_node);

        num_nodes = std::max(std::max(from_node, to_node), num_nodes);
    }
    // assume the node ids start from zero
    ++num_nodes;

    for (uint_fast32_t node = 0; node < num_nodes; ++node) {
        const bool isdangling = out_edges.count(node) == 0;
        if (isdangling) {
            dangling_nodes.insert(node);
        }
    }

    file.close();
}


CSC::CSC(const Graph& graph)
{
    num_rows = num_cols = graph.num_nodes;

    uint_fast32_t num_nonzero_values = 0;
    ia.push_back(num_nonzero_values);

    for (uint_fast32_t to_node = 0; to_node < graph.num_nodes; ++to_node) {
        const uint_fast32_t indegree = graph.in_edges.count(to_node) > 0 ?
                                       graph.in_edges.at(to_node).size() : 0;
        if (indegree > 0) {
            for (uint_fast32_t from_node: graph.in_edges.at(to_node)) {
                a.push_back(1.0f/indegree);
                ja.push_back(from_node);
            }
            num_nonzero_values += indegree;
        }
        ia.push_back(num_nonzero_values);
    }
}


arma::fvec CSR::operator*(const arma::fvec& vec) const
{
    assert(num_cols == vec.size());

    arma::fvec res(num_rows, arma::fill::zeros);
    for (uint_fast32_t i = 0; i < num_rows; ++i) {
        for (uint_fast32_t k = ia[i]; k < ia[i+1]; ++k) {
            res[i] += a[k] * vec[ja[k]];
        }
    }
    return res;
}

std::vector<std::pair<uint_fast32_t, CSR>> CSR::split(uint_fast32_t n) const
{
    // TODO clean
    assert(0 < n and n <= num_rows);

    // compute maximum size of each submatrix
    // note that the last submatrix can have a smaller size than the others
    const uint_fast32_t size = ceil(float(num_rows)/n);

    int totoff = 0;
    std::vector<std::pair<uint_fast32_t, CSR>> csrs;
    uint_fast32_t i = 1, j = 0;
    uint_fast32_t offset = 0;
    do {
        auto subcsr = CSR();

        subcsr.ia.push_back(0);
        for (uint_fast32_t k = 0; i < ia.size() && k < size; ++k) {
            subcsr.ia.push_back(ia[i] - offset);
            ++i;
        }
        offset += subcsr.ia.back();
        assert(((csrs.size() < n-1) and subcsr.ia.size() == size+1) or
               ((csrs.size() == n-1) and subcsr.ia.size() <= size+1));

        for (uint_fast32_t k = 0; k < subcsr.ia.back(); ++k) {
            subcsr.a.push_back(a[j]);
            subcsr.ja.push_back(ja[j]);
            ++j;
        }

        subcsr.num_rows = subcsr.ia.size()-1;
        subcsr.num_cols = num_cols;
        csrs.push_back(std::make_pair(totoff, subcsr));
        totoff += size;
    }
    while (csrs.size() < n);

    assert(i == ia.size());
    assert(j == a.size() and j == ja.size());
    return csrs;
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
