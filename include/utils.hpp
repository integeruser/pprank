#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "armadillo"


struct Graph {
    uint_fast32_t num_nodes;
    std::map<uint_fast32_t, std::set<uint_fast32_t>> in_edges;
    std::map<uint_fast32_t, uint_fast32_t> outdegrees;
    std::set<uint_fast32_t> dangling_nodes;

    Graph(const std::string&);
};


struct CSC {
    uint_fast32_t num_rows, num_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia;
    std::vector<uint_fast32_t> ja;
    std::vector<uint_fast32_t> dangling_nodes;

    CSC() {}
    CSC(const Graph&);
    CSC(std::ifstream&);

    void to_file(std::ofstream&) const;
};

struct CSR {
    uint_fast32_t num_rows, num_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia;
    std::vector<uint_fast32_t> ja;
    std::vector<uint_fast32_t> dangling_nodes;

    CSR() {}
    CSR(const arma::fmat&);

    arma::fvec operator*(const arma::fvec&) const;

    std::vector<std::pair<uint_fast32_t, CSR>> split(uint_fast32_t) const;
};

CSR transpose(const CSC&);


#endif
