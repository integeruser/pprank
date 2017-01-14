#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "armadillo"


struct Graph {
    uint_fast32_t num_nodes;
    std::map<uint_fast32_t, std::set<uint_fast32_t>> edges;

    Graph(const std::string&);
};


arma::sp_fmat to_adjacency_mat(const Graph&);


#endif
