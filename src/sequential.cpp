#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>

#include "utils.hpp"

#include "armadillo"

using hrc = std::chrono::high_resolution_clock;


std::tuple<uint_fast32_t, pprank_vec_t> pagerank(const TCSR& A, const pprank_t tol)
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

        pprank_vec_t At_dot_p = A.tdot(p);
        At_dot_p += arma::sum(p(dangling_nodes))/N * ones;

        p_new = (1.0-d)/N * ones + d * At_dot_p;
    }
    while (arma::norm(p_new-p, 1) >= tol);
    return std::make_tuple(iterations, p_new);
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

    // build the sparse transition matrix
    std::cout << "[*] Building the sparse transition matrix..." << std::flush;
    start_time = hrc::now();

    const TCSR tcsr = TCSR(filename);
    assert(tcsr.num_rows == tcsr.num_cols);

    end_time = hrc::now();
    duration = end_time-start_time;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << duration.count() << " s]" << std::endl;
    std::cout << "        Nodes:      " << tcsr.num_rows << std::endl;
    std::cout << "        Edges:      " << tcsr.a.size() << std::endl;
    std::cout << "        Dangling:   " << tcsr.dangling_nodes.size() << std::endl;
    ////////////////////////////////////////////////////////////////////////////

    const pprank_t tol = 1e-6;

    // compute PageRanks
    std::cout << std::fixed << std::scientific;
    std::cout << "[*] Computing PageRanks (tol=" << tol << ")..." << std::flush;
    start_time = hrc::now();

    uint_fast32_t iterations;
    pprank_vec_t ranks;
    std::tie(iterations, ranks) = pagerank(tcsr, tol);

    end_time = hrc::now();
    duration = end_time-start_time;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << iterations << " iterations - " << duration.count() << " s]" << std::endl;
    ////////////////////////////////////////////////////////////////////////////

    // write PageRanks to file
    std::cout << "[*] Writing PageRanks to file..." << std::flush;
    start_time = hrc::now();

    std::ofstream outfile("PageRanks-" + std::to_string(tcsr.num_rows) + "-" + std::to_string(tcsr.a.size()) + ".txt");
    outfile << std::fixed << std::scientific;
    for (uint_fast32_t node = 0; node < ranks.size(); ++node) {
        outfile << std::setfill('0') << std::setw(9) << node << ": " << ranks[node] << std::endl;
    }
    outfile.close();

    end_time = hrc::now();
    duration = end_time-start_time;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << duration.count() << " s]" << std::endl;

    return EXIT_SUCCESS;
}
