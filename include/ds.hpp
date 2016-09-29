#ifndef DS_HPP
#define DS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "armadillo"


struct Graph {
    std::map<uint_fast32_t, std::set<uint_fast32_t>> edges;

    Graph(const std::string&);
};


struct CSR {
    size_t n_rows, n_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia;
    std::vector<uint_fast32_t> ja;

    CSR() {}
    CSR(const arma::mat&);
    CSR(const Graph&);
    CSR(const std::string&);

    arma::vec operator*(const arma::vec&);

    std::vector<CSR> split(size_t) const;

    void to_file();
};


#endif
