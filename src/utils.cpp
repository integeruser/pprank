#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
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
    // checks for duplicate edges are NOT performed
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

        outedges[from_node].insert(to_node);

        num_nodes = std::max(std::max(from_node, to_node), num_nodes);
    }
    // assume the node ids start from zero
    ++num_nodes;

    for (uint_fast32_t node = 0; node < num_nodes; ++node) {
        const uint_fast32_t outdegree = outedges.count(node) > 0 ?
                                        outedges.at(node).size() : 0;
        const bool isdangling = outdegree == 0;
        if (isdangling) {
            dangling_nodes.insert(node);
        }
    }

    file.close();
}


CSR::CSR()
{
}

CSR::CSR(const Graph& graph)
{
    num_rows = num_cols = graph.num_nodes;
    dangling_nodes = std::vector<uint_fast32_t>(graph.dangling_nodes.cbegin(), graph.dangling_nodes.cend());

    uint_fast32_t num_nonzero_values = 0;
    ia.push_back(num_nonzero_values);

    for (uint_fast32_t from_node = 0; from_node < graph.num_nodes; ++from_node) {
        const uint_fast32_t outdegree = graph.outedges.count(from_node) > 0 ?
                                        graph.outedges.at(from_node).size() : 0;
        if (outdegree > 0) {
            for (uint_fast32_t to_node: graph.outedges.at(from_node)) {
                a.push_back(1.0f/outdegree);
                ja.push_back(to_node);
            }
            num_nonzero_values += outdegree;
        }
        ia.push_back(num_nonzero_values);
    }
}

CSR::CSR(const std::string& filename)
{
    // std::ifstream file(filename, std::ios::in);

    // uint_fast32_t num_nodes = 281903;
    // uint_fast32_t num_edges = 2312497;

    // uint_fast32_t num_nonzero_values = 0;
    // ia.push_back(num_nonzero_values);

    // uint_fast32_t curr_node = 0;
    // std::set<uint_fast32_t> curr_outedges = std::set<uint_fast32_t>();

    // std::string line;
    // while (std::getline(file, line)) {
    //     std::istringstream iss(line);
    //     uint_fast32_t from_node, to_node;
    //     if (!(iss >> from_node >> to_node)) {
    //         // skip malformed lines
    //         std::cout << "malformed " << line << std::endl;
    //         continue;
    //     }

    //     if (from_node == curr_node) {
    //         curr_outedges.insert(to_node);
    //     }
    //     else {
    //         const uint_fast32_t curr_isdangling = curr_outedges.size() == 0;
    //         if (curr_isdangling) {
    //             dangling_nodes.push_back(curr_node);
    //         }
    //         else {
    //             for (uint_fast32_t to_node: curr_outedges) {
    //                 a.push_back(1.0f/curr_outedges.size());
    //                 ja.push_back(to_node);
    //                 ++num_nonzero_values;
    //             }
    //         }
    //         ia.push_back(num_nonzero_values);

    //         curr_node = from_node;
    //         curr_outedges = std::set<uint_fast32_t> {to_node};
    //     }

    //     // num_nodes = std::max(std::max(from_node, to_node), num_nodes);
    // }

    // const uint_fast32_t curr_isdangling = curr_outedges.size() == 0;
    // if (curr_isdangling) {
    //     dangling_nodes.push_back(curr_node);
    // }
    // else {
    //     for (uint_fast32_t to_node: curr_outedges) {
    //         a.push_back(1.0f/curr_outedges.size());
    //         ja.push_back(to_node);
    //     }
    //     ++curr_node;
    // }
    // num_nonzero_values += curr_outedges.size();
    // ia.push_back(num_nonzero_values);

    // // assume the node ids start from zero
    // // ++num_nodes;
    // // std::cout << "num nodes" << num_nodes << std::endl;
    // num_rows = num_cols = num_nodes;

    // for (; curr_node < num_nodes; ++curr_node) {
    //     // std::cout << curr_node << ", ";
    //     dangling_nodes.push_back(curr_node);
    //     ia.push_back(num_nonzero_values);
    // }

    // // std::cout << "a" << a << std::endl;
    // // std::cout << "ia" << ia << std::endl;
    // // std::cout << "ja" << ja << std::endl;
    // // std::cout << "dangling_nodes" << dangling_nodes << std::endl;

    // file.close();
}

arma::fvec CSR::dot_transposed(const arma::fvec& vec) const
{
    arma::fvec res(num_rows, arma::fill::zeros);
    for (uint_fast32_t i = 0; i < num_rows; ++i) {
        for (uint_fast32_t k = ia[i]; k < ia[i+1]; ++k) {
            res[ja[k]] += a[k] * vec[i];
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
