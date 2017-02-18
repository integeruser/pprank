#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"

using hrc = std::chrono::high_resolution_clock;


std::pair<uint_fast32_t, pprank_vec_t> pagerank(const TCSR& A, const pprank_t tol)
{
    assert(A.num_rows == A.num_cols);

    // initialization
    const uint_fast32_t N = A.num_rows;
    const pprank_t d = 0.85;
    const pprank_vec_t ones(N, arma::fill::ones);
    const arma::uvec dangling_nodes = arma::conv_to<arma::uvec>::from(A.dangling_nodes);

    pprank_vec_t p(N), p_new(N);
    p_new.fill(1.0/N);

    // ranks computation
    uint_fast32_t iterations = 0;
    do {
        ++iterations;
        p = p_new;

        const pprank_vec_t dangling = arma::sum(p(dangling_nodes))/N * ones;
        p_new = (1.0-d)/N * ones + d * (A.tdot(p) + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);
    return std::make_pair(iterations, p_new);
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: sequential file" << std::endl;
        return EXIT_FAILURE;
    }
    const char* filename = argv[1];

    hrc::time_point start_time, end_time;
    std::chrono::duration<pprank_t> duration;

    // build the TCSR matrix
    std::cout << "[*] Building the sparse transition matrix..." << std::flush;
    start_time = hrc::now();

    const TCSR tcsr = TCSR(filename);
    assert(tcsr.num_rows == tcsr.num_cols);

    end_time = hrc::now();
    duration = end_time-start_time;
    std::cout << "[" << duration.count() << " s]" << std::endl;
    std::cout << "        Nodes:      " << tcsr.num_rows << std::endl;
    std::cout << "        Edges:      " << tcsr.a.size() << std::endl;
    std::cout << "        Dangling:   " << tcsr.dangling_nodes.size() << std::endl;

    const pprank_t tol = 1e-6;

    // compute PageRanks
    std::cout << "[*] Computing PageRanks (tol=" << tol << ")..." << std::flush;
    start_time = hrc::now();

    const std::pair<uint_fast32_t, pprank_vec_t> results = pagerank(tcsr, tol);
    const uint_fast32_t iterations = results.first;
    const pprank_vec_t ranks = results.second;

    end_time = hrc::now();
    duration = end_time-start_time;
    std::cout << "[" << iterations << " iterations - " << duration.count() << " s]" << std::endl;

    // write PageRanks to file
    std::cout << "[*] Writing PageRanks to file..." << std::endl;

    std::ofstream outfile("PageRanks-" + std::to_string(tcsr.num_rows) + "-" + std::to_string(tcsr.a.size()) + ".txt");
    outfile << std::fixed << std::scientific;
    for (uint_fast32_t node = 0; node < ranks.size(); ++node) {
        outfile << std::setfill('0') << std::setw(9) << node << ": " << ranks[node] << std::endl;
    }
    outfile.close();

    return EXIT_SUCCESS;
}
