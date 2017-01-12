#ifndef DS_HPP
#define DS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "armadillo"

using NodeIndex = uint_fast32_t;


struct Graph {
    std::map<NodeIndex, std::set<NodeIndex>> edges;

    Graph(const std::string&);

    arma::mat to_dense() const;
};


struct CSR {
    std::size_t n_rows, n_cols;
    std::vector<float> a;
    std::vector<NodeIndex> ia;
    std::vector<NodeIndex> ja;

    CSR() {}
    CSR(const arma::mat&);
    CSR(const Graph&);
    CSR(const std::string&);

    arma::vec operator*(const arma::vec&);

    std::vector<CSR> split(std::size_t) const;

    void to_file();
};


#endif
