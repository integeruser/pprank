#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "armadillo"

#ifdef ACCURATE
using pprank_t = double;
using pprank_vec_t = arma::vec;
#define PPRANK_MPI_T MPI_DOUBLE
#else
using pprank_t = float;
using pprank_vec_t = arma::fvec;
#define PPRANK_MPI_T MPI_FLOAT
#endif


struct TCSR {
    uint_fast32_t num_rows, num_cols;
    std::vector<pprank_t> a;
    std::vector<uint_fast32_t> ia, ja;
    std::vector<uint_fast32_t> dangling_nodes;

    TCSR();
    TCSR(const std::string&);

    pprank_vec_t tdot(const pprank_vec_t&) const;

    std::tuple<std::vector<uint_fast32_t>, std::vector<uint_fast32_t>, std::vector<TCSR>> split(uint_fast32_t) const;
};


#endif
