#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const CSC& A, const float tol)
{
    // initialization
    const uint_fast32_t N = A.num_rows;

    const CSR At = transpose(A);

    arma::fvec p, p_new(N);
    p_new.fill(1.0f/N);

    const arma::uvec dangling_nodes = arma::conv_to<arma::uvec>::from(At.dangling_nodes);

    const arma::fvec ones(N, arma::fill::ones);
    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    std::cout << "[*] Computing PageRank (tol=" << tol << ")..." << std::endl;
    do {
        ++iterations;

        p = p_new;

        const arma::fvec dangling = 1.0f/N * arma::sum(p(dangling_nodes)) * ones;
        p_new = (1-d)/N * ones + d * (At*p + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < N; ++node) {
        ranks[node] = p[node];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char const *argv[])
{
    if (!(argc == 2 || argc == 3)) {
        std::cerr << "Usage: sequential file [tol]" << std::endl;
        return EXIT_FAILURE;
    }

    const char* file = argv[1];
    float tol = 1e-6f;
    if (argc == 3) {
        tol = std::atof(argv[2]);
    }

    std::ifstream infile(file, std::ios::in);
    const CSC csc = CSC(infile);
    infile.close();

    const auto results = pagerank(csc, tol);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "[*] Ranks (after " << iterations << " iterations):" << std::endl;
    for (const auto pair: ranks) {
        const uint_fast32_t node = pair.first;
        const float rank = pair.second;
        std::cout << "        " << std::setfill('0') << std::setw(9) << node << ": " << rank << std::endl;
    }

    return EXIT_SUCCESS;
}
