#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const TCSR& A, const float tol)
{
    assert(A.num_rows == A.num_cols);

    // initialization
    const uint_fast32_t N = A.num_rows;

    arma::fvec p, p_new(N);
    p_new.fill(1.0f/N);

    const arma::uvec dangling_nodes = arma::conv_to<arma::uvec>::from(A.dangling_nodes);

    const arma::fvec ones(N, arma::fill::ones);
    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    do {
        ++iterations;

        p = p_new;

        const arma::fvec dangling = 1.0f/N * arma::sum(p(dangling_nodes)) * ones;
        p_new = (1-d)/N * ones + d * (A.tdot(p) + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < N; ++node) {
        ranks[node] = p[node];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char *argv[])
{
    if (!(argc == 2 || argc == 3)) {
        std::cerr << "Usage: sequential file [tol]" << std::endl;
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    float tol = 1e-6f;
    if (argc == 3) {
        tol = std::atof(argv[2]);
    }

    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<float> duration;

    std::cout << "[*] Building TCSR transition matrix..." << std::flush;
    start_time = std::chrono::high_resolution_clock::now();
    const TCSR tcsr = TCSR(filename);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time-start_time;
    std::cout << "[" << duration.count() << " s]" << std::endl;

    assert(tcsr.num_rows == tcsr.num_cols);
    std::cout << "        Nodes:      " << tcsr.num_rows << std::endl;
    std::cout << "        Edges:      " << tcsr.a.size() << std::endl;
    std::cout << "        Dangling:   " << tcsr.dangling_nodes.size() << std::endl;

    std::cout << "[*] Computing PageRank (tol=" << tol << ")..." << std::flush;
    start_time = std::chrono::high_resolution_clock::now();
    const auto results = pagerank(tcsr, tol);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time-start_time;
    std::cout << "[" << duration.count() << " s]" << std::endl;

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
