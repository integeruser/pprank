#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include <tuple>
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
    //  - filename must contain the number of nodes and the number of edges of the graph in the form "(\d+)-(\d+)"
    //  - lines starting with '#' are ignored
    //  - node ids are zero-based
    //  - no duplicate edges
    //  - edges are ordered by source node id
    //  - the file ends with a newline

    std::ifstream file(filename);

    // parse header
    std::regex header("(?:([0-9]+)-([0-9]+))(?!.*[0-9]*-[0-9]*)");
    std::smatch matches;
    std::regex_search(filename, matches, header);
    if (matches.size() != 3) {
        std::cerr << "[!] Filename not compliant!" << std::endl;
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
            // skip lines starting with '#'
            if (line[0] == '#') { continue; }

            // each line represents a directed edge between two nodes
            uint_fast32_t from_node, to_node;
            read_edge(line, from_node, to_node);

            if (from_node != curr_node) {
                assert(curr_node < from_node);
                // all outedges of curr_node have been found
                for (uint_fast32_t i = 0; i < curr_outdegree; ++i) {
                    a.push_back(1.0/curr_outdegree);
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
        a.push_back(1.0/curr_outdegree);
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

pprank_vec_t TCSR::tdot(const pprank_vec_t& vec) const
{
    // compute a matrix-vector product with the matrix transposed
    pprank_vec_t res(num_cols, arma::fill::zeros);
    for (uint_fast32_t i = 0; i < num_rows; ++i) {
        for (uint_fast32_t k = ia[i]; k < ia[i+1]; ++k) {
            res[ja[k]] += a[k] * vec[i];
        }
    }
    return res;
}

std::tuple<std::vector<uint_fast32_t>, std::vector<uint_fast32_t>, std::vector<TCSR>> TCSR::split(uint_fast32_t n) const
{
    // split the matrix by rows into n submatrices
    assert(0 < n and n <= num_rows);

    std::vector<uint_fast32_t> displacements, rows;
    std::vector<TCSR> tcsrs;

    // compute maximum size of each submatrix
    // note that the last one can have fewer rows than the others
    const uint_fast32_t num_rows_sub = std::ceil(((pprank_t) num_rows)/n);

    uint_fast32_t i = 1, j = 0;
    uint_fast32_t start = 0, offset = 0;
    do {
        TCSR tcsr_sub = TCSR();

        tcsr_sub.ia.push_back(0);

        for (uint_fast32_t k = 0; i < ia.size() and k < num_rows_sub; ++i, ++k) {
            tcsr_sub.ia.push_back(ia[i]-start);
        }
        start += tcsr_sub.ia.back();
        assert(((tcsrs.size() < n-1) and tcsr_sub.ia.size() == num_rows_sub+1) or
               ((tcsrs.size() == n-1) and tcsr_sub.ia.size() <= num_rows_sub+1));

        for (uint_fast32_t k = 0; k < tcsr_sub.ia.back(); ++j, ++k) {
            tcsr_sub.a.push_back(a[j]);
            tcsr_sub.ja.push_back(ja[j]);
        }

        tcsr_sub.num_rows = tcsr_sub.ia.size()-1;
        tcsr_sub.num_cols = num_cols;

        displacements.push_back(offset);
        rows.push_back(tcsr_sub.num_rows);
        tcsrs.push_back(tcsr_sub);
        offset += num_rows_sub;
    }
    while (tcsrs.size() < n);

    assert(i == ia.size());
    assert(j == a.size() and j == ja.size());
    return std::make_tuple(displacements, rows, tcsrs);
}
