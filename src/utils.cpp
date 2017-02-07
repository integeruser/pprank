#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "utils.hpp"

#include "armadillo"


CSR::CSR()
{
}

CSR::CSR(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in);

    std::string line;

    // parse header
    std::regex header("# Nodes: ([0-9]+) Edges: ([0-9]+)");
    std::smatch matches;
    std::getline(file, line);
    std::regex_search(line, matches, header);
    assert(matches.size() == 3);
    const uint_fast32_t num_nodes = std::stoul(matches[1].str());
    const uint_fast32_t num_edges = std::stoul(matches[2].str());

    // assumptions:
    //  - node ids start from zero
    //  - each line of the file represent an edge from a source node to a destination node
    //  - no duplicate edges
    //  - lines are ordered by source node id
    //  - lines that start with "#" are comments

    a.reserve(num_edges);
    ja.reserve(num_edges);

    uint_fast32_t num_nonzero_values = 0;
    ia.push_back(num_nonzero_values);

    uint_fast32_t num_nodes_read = 0;

    uint_fast32_t curr_node = 0;
    std::set<uint_fast32_t> curr_outedges;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        uint_fast32_t from_node, to_node;
        if (!(iss >> from_node >> to_node)) {
            // skip comments and malformed lines
            if (line[0] != '#') {
                std::cerr << "Malformed line: \"" << line << "\"" << std::endl;
            }
            continue;
        }

        if (from_node == curr_node) {
            curr_outedges.insert(to_node);
        }
        else {
            // add a row to the transition matrix
            const uint_fast32_t curr_outdegree = curr_outedges.size();
            for (uint_fast32_t to_node: curr_outedges) {
                a.push_back(1.0f/curr_outdegree);
                ja.push_back(to_node);
            }
            num_nonzero_values += curr_outdegree;
            ia.push_back(num_nonzero_values);

            // add dangling nodes until needed
            for (uint_fast32_t node = curr_node+1; node < from_node; ++node) {
                dangling_nodes.push_back(node);
                ia.push_back(num_nonzero_values);
            }

            curr_node = from_node;
            curr_outedges = {to_node};
        }

        num_nodes_read = std::max(std::max(from_node, to_node), num_nodes_read);
    }
    // assume the node ids start from zero
    ++num_nodes_read;
    assert(num_nodes_read == num_nodes);

    num_rows = num_cols = num_nodes_read;

    // add a last row to the transition matrix
    const uint_fast32_t curr_outdegree = curr_outedges.size();
    for (uint_fast32_t to_node: curr_outedges) {
        a.push_back(1.0f/curr_outdegree);
        ja.push_back(to_node);
    }
    num_nonzero_values += curr_outedges.size();
    ia.push_back(num_nonzero_values);

    // add dangling nodes until needed
    for (uint_fast32_t node = curr_node+1; node < num_nodes_read; ++node) {
        dangling_nodes.push_back(node);
        ia.push_back(num_nonzero_values);
    }

    file.close();
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
