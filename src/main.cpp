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

#include "mpi.h"

#define MASTER 0

int rank;
int num_processes;


std::pair<uint_fast32_t, arma::fvec> pagerank(const TCSR& A, const float tol)
{
    assert(A.num_rows == A.num_cols);

    // initialization
    const uint_fast32_t N = A.num_rows;

    arma::fvec p, p_new(N);
    p_new.fill(1.0f/N);

    const arma::uvec dangling_nodes = arma::conv_to<arma::uvec>::from(A.dangling_nodes);

    const arma::fvec ones(N, arma::fill::ones);
    const float d = 0.85f;

    // partition the matrix in blocks of rows
    const auto blocks = A.split(num_processes);
    std::vector<int> blocks_displacements, blocks_num_rows;
    for (const auto block: blocks) {
        blocks_displacements.push_back(block.first);
        blocks_num_rows.push_back(block.second.num_rows);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ranks computation
    uint_fast32_t iterations = 0;
    do {
        ++iterations;

        p = p_new;

        // each process computes the matrix-vector product only for its block of rows
        const TCSR A_block = blocks[rank].second;
        const arma::fvec subvec = p.subvec(blocks_displacements[rank], blocks_displacements[rank]+blocks_num_rows[rank]-1);
        const arma::fvec prod_block = A_block.tdot(subvec);

        // gather the results of the matrix-vector products
        arma::fvec prod(N);
        MPI_Allreduce(prod_block.memptr(), prod.memptr(),
                      N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        const arma::fvec dangling = 1.0f/N * arma::sum(p(dangling_nodes)) * ones;
        p_new = (1-d)/N * ones + d * (prod + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);
    p = p_new;

    return std::make_pair(iterations, p);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (!(argc == 2 || argc == 3)) {
        if (rank == MASTER) { std::cerr << "Usage: sequential file [tol]" << std::endl; }
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    float tol = 1e-6f;
    if (argc == 3) {
        tol = std::atof(argv[2]);
    }

    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<float> duration;

    if (rank == MASTER) { std::cout << "[*] Building TCSR transition matrix..." << std::flush; }
    start_time = std::chrono::high_resolution_clock::now();
    const TCSR tcsr = TCSR(filename);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time-start_time;
    if (rank == MASTER) { std::cout << "[" << duration.count() << " s]" << std::endl; }

    assert(tcsr.num_rows == tcsr.num_cols);
    if (rank == MASTER) {
        std::cout << "        Nodes:      " << tcsr.num_rows << std::endl;
        std::cout << "        Edges:      " << tcsr.a.size() << std::endl;
        std::cout << "        Dangling:   " << tcsr.dangling_nodes.size() << std::endl;
    }

    if (rank == MASTER) { std::cout << "[*] Computing PageRank (tol=" << tol << ")..." << std::flush; }
    start_time = std::chrono::high_resolution_clock::now();
    const auto results = pagerank(tcsr, tol);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time-start_time;
    if (rank == MASTER) { std::cout << "[" << duration.count() << " s]" << std::endl; }

    if (rank == MASTER) {
        const auto iterations = results.first;
        const auto ranks = results.second;
        std::cout << "[*] Ranks (after " << iterations << " iterations):" << std::endl;
        for (uint_fast32_t node = 0; node < ranks.size(); ++node) {
            std::cout << "        " << std::setfill('0') << std::setw(9) << node << ": " << ranks[node] << std::endl;
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
