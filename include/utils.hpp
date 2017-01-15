#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "armadillo"


struct Graph {
    uint_fast32_t num_nodes;
    std::map<uint_fast32_t, std::set<uint_fast32_t>> in_edges;
    std::map<uint_fast32_t, std::set<uint_fast32_t>> out_edges;

    Graph(const std::string&);
};


struct CSC {
    uint_fast32_t num_rows, num_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia;
    std::vector<uint_fast32_t> ja;

    CSC() {}
    CSC(const Graph&);
};

struct CSR {
    uint_fast32_t num_rows, num_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia;
    std::vector<uint_fast32_t> ja;

    CSR() {}
    CSR(const Graph&);

    arma::fvec operator*(const arma::fvec&) const;

    std::vector<std::pair<uint_fast32_t, CSR>> split(uint_fast32_t) const;
};

CSR transpose(const CSC&);


#endif
