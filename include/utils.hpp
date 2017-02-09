#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "armadillo"


struct TCSR {
    uint_fast32_t num_rows, num_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia, ja;
    std::vector<uint_fast32_t> dangling_nodes;

    TCSR();
    TCSR(const std::string&);

    arma::fvec tdot(const arma::fvec&) const;

    std::vector<std::pair<uint_fast32_t, TCSR>> split(uint_fast32_t) const;
};


#endif
