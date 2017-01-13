#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<size_t, std::map<uint_fast32_t, float>> pagerank(const arma::sp_fmat& A)
{
    // initialization
    assert(A.n_rows == A.n_cols);
    const auto At = A.t();

    const size_t n = A.n_rows;

    arma::fvec p(n), p_prev;
    p.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const auto d = 0.85f;

    // ranks computation
    size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;
        p = (1-d)/n * ones + d * (At*p);
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (size_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    const auto filename = argv[1];
    const auto adjacency_mat = load(filename);

    const auto results = pagerank(adjacency_mat);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;

    return EXIT_SUCCESS;
}
