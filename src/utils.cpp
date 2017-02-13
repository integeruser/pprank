#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <string.h>

#include "utils.hpp"

#include "armadillo"


inline void read_edge(char* line, uint_fast32_t& from_node, uint_fast32_t& to_node)
{
    char* endptr;
    const uint_fast32_t base = 10;
    from_node = std::strtoul(line, &endptr, base);
    to_node = std::strtoul(endptr, nullptr, base);
}


TCSR::TCSR()
{
}

TCSR::TCSR(const std::string& filename)
{
    // construct a transition (sparse) matrix from a file
    // each line of the file represents an edge from a source node to a destination node

    // assumptions:
    //  - first line of file is the header "Nodes: <num_nodes> Edges: <num_edges>"
    //  - node ids are zero-based
    //  - no duplicate edges
    //  - edges are ordered by source node id
    //  - the file ends with a newline

    std::ifstream file(filename);

    // parse header
    std::regex re_header("# Nodes: ([0-9]+) Edges: ([0-9]+)");
    std::smatch matches;
    std::string header;
    std::getline(file, header);
    std::regex_search(header, matches, re_header);
    if (matches.size() != 3) {
        std::cerr << "[Err] Malformed header!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const uint_fast32_t num_nodes = std::stoul(matches[1].str());
    const uint_fast32_t num_edges = std::stoul(matches[2].str());

    num_rows = num_cols = num_nodes;
    a.reserve(num_edges);
    ja.reserve(num_edges);

    uint_fast32_t curr_node = 0, curr_outdegree = 0;

    uint_fast32_t num_nonzero_values = 0;
    ia.push_back(num_nonzero_values);

    // read edges line by line
    const uint_fast32_t BUF_SIZE = 1024*16;
    char buf[BUF_SIZE];
    while (file) {
        // read from file into a buffer
        file.read(buf, BUF_SIZE);
        const ssize_t bytes_read = file.gcount();

        char *line, *newline_char;
        for (line = buf; (newline_char = (char*) memchr(line, '\n', (buf+bytes_read)-line)); line = newline_char+1) {
            // each line represents a directed edge between two nodes
            uint_fast32_t from_node, to_node;
            read_edge(line, from_node, to_node);

            if (from_node != curr_node) {
                assert(curr_node < from_node);
                // all outedges of curr_node have been found
                for (uint_fast32_t i = 0; i < curr_outdegree; ++i) {
                    a.push_back(1.0f/curr_outdegree);
                }
                num_nonzero_values += curr_outdegree;
                ia.push_back(num_nonzero_values);

                // if needed, add dangling nodes
                for (uint_fast32_t node = curr_node+1; node < from_node; ++node) {
                    dangling_nodes.push_back(node);
                    ia.push_back(num_nonzero_values);
                }

                curr_node = from_node;
                curr_outdegree = 0;
            }
            ++curr_outdegree;
            ja.push_back(to_node);
        }
        const ptrdiff_t bytes_consumed = line-buf;
        file.seekg(-(bytes_read-bytes_consumed), std::ios_base::cur);
    }

    // add outedges of the last parsed node
    for (uint_fast32_t i = 0; i < curr_outdegree; ++i) {
        a.push_back(1.0f/curr_outdegree);
    }
    num_nonzero_values += curr_outdegree;
    ia.push_back(num_nonzero_values);
    assert(num_nonzero_values == num_edges);

    // if needed, add dangling nodes
    for (uint_fast32_t node = curr_node+1; node < num_nodes; ++node) {
        dangling_nodes.push_back(node);
        ia.push_back(num_nonzero_values);
    }

    file.close();
}

arma::fvec TCSR::tdot(const arma::fvec& vec) const
{
    // compute a matrix-vector product with the matrix transposed
    arma::fvec res(num_rows, arma::fill::zeros);
    for (uint_fast32_t i = 0; i < num_rows; ++i) {
        for (uint_fast32_t k = ia[i]; k < ia[i+1]; ++k) {
            res[ja[k]] += a[k] * vec[i];
        }
    }
    return res;
}

std::vector<std::pair<uint_fast32_t, TCSR>> TCSR::split(uint_fast32_t n) const
{
    // TODO clean
    assert(0 < n and n <= num_rows);

    // compute maximum size of each submatrix
    // note that the last submatrix can have a smaller size than the others
    const uint_fast32_t size = ceil(float(num_rows)/n);

    int totoff = 0;
    std::vector<std::pair<uint_fast32_t, TCSR>> tcsrs;
    uint_fast32_t i = 1, j = 0;
    uint_fast32_t offset = 0;
    do {
        auto subtcsr = TCSR();

        subtcsr.ia.push_back(0);
        for (uint_fast32_t k = 0; i < ia.size() && k < size; ++k) {
            subtcsr.ia.push_back(ia[i] - offset);
            ++i;
        }
        offset += subtcsr.ia.back();
        assert(((tcsrs.size() < n-1) and subtcsr.ia.size() == size+1) or
               ((tcsrs.size() == n-1) and subtcsr.ia.size() <= size+1));

        for (uint_fast32_t k = 0; k < subtcsr.ia.back(); ++k) {
            subtcsr.a.push_back(a[j]);
            subtcsr.ja.push_back(ja[j]);
            ++j;
        }

        subtcsr.num_rows = subtcsr.ia.size()-1;
        subtcsr.num_cols = num_cols;
        tcsrs.push_back(std::make_pair(totoff, subtcsr));
        totoff += size;
    }
    while (tcsrs.size() < n);

    assert(i == ia.size());
    assert(j == a.size() and j == ja.size());
    return tcsrs;
}
