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

#include "mpi.h"

#define MASTER 0

using hrc = std::chrono::high_resolution_clock;

int rank;
int num_processes;


std::tuple<uint_fast32_t, double, double, pprank_vec_t> pagerank(const TCSR& A, const pprank_t tol)
{
    assert(A.num_rows == A.num_cols);

    // initialization
    const uint_fast32_t N = A.num_rows;
    const pprank_t d = 0.85;
    const pprank_vec_t ones(N, arma::fill::ones);
    const arma::uvec dangling_nodes = arma::conv_to<arma::uvec>::from(A.dangling_nodes);

    pprank_vec_t p(N), p_new(N);
    p_new.fill(1.0/N);

    // partition the matrix in blocks of rows
    std::vector<uint_fast32_t> displacements, sizes;
    std::vector<TCSR> tcsrs;
    std::tie(displacements, sizes, tcsrs) = A.split(num_processes);

    MPI_Barrier(MPI_COMM_WORLD);

    // ranks computation
    double start_time, work_time = 0.0, netw_time = 0.0;
    uint_fast32_t iterations = 0;
    do {
        ++iterations;
        p = p_new;

        // each node calculates a partial result of the matrix-vector product
        start_time = MPI_Wtime();

        const TCSR A_sub = tcsrs[rank];
        const pprank_vec_t At_dot_p_sub = A_sub.tdot(p);

        work_time += MPI_Wtime()-start_time;
        ////////////////////////////////////////////////////////////////////////

        // collect and sum the partial results of all the nodes
        // each node must receive the contributions from all the others (very heavy!)
        start_time = MPI_Wtime();

        pprank_vec_t At_dot_p(N);
        MPI_Allreduce(At_dot_p_sub.memptr(), At_dot_p.memptr(), N, PPRANK_MPI_T, MPI_SUM, MPI_COMM_WORLD);

        netw_time += MPI_Wtime()-start_time;
        ////////////////////////////////////////////////////////////////////////

        // update PageRanks
        start_time = MPI_Wtime();

        At_dot_p += arma::sum(p(dangling_nodes))/N * ones;

        p_new = (1.0-d)/N * ones + d * At_dot_p;

        work_time += MPI_Wtime()-start_time;
    }
    while (arma::norm(p_new-p, 1) >= tol);
    return std::make_tuple(iterations, work_time, netw_time, p_new);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (argc != 2) {
        std::cerr << "Usage: pprank file" << std::endl;
        return EXIT_FAILURE;
    }
    const char* filename = argv[1];

    hrc::time_point start_time, end_time;
    std::chrono::duration<pprank_t> duration;

    // build the sparse transition matrix
    if (rank == MASTER) {
        std::cout << "[*] Building the sparse transition matrix..." << std::flush;
        start_time = hrc::now();
    }

    const TCSR tcsr = TCSR(filename);
    assert(tcsr.num_rows == tcsr.num_cols);

    if (rank == MASTER) {
        end_time = hrc::now();
        duration = end_time-start_time;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << duration.count() << " s]" << std::endl;
        std::cout << "        Nodes:      " << tcsr.num_rows << std::endl;
        std::cout << "        Edges:      " << tcsr.a.size() << std::endl;
        std::cout << "        Dangling:   " << tcsr.dangling_nodes.size() << std::endl;
    }
    ////////////////////////////////////////////////////////////////////////////

    const pprank_t tol = 1e-6;

    // compute PageRanks
    if (rank == MASTER) {
        std::cout << std::fixed << std::scientific;
        std::cout << "[*] Computing PageRanks (tol=" << tol << ")..." << std::flush;
        start_time = hrc::now();
    }

    uint_fast32_t iterations;
    double work_time, netw_time;
    pprank_vec_t ranks;
    std::tie(iterations, work_time, netw_time, ranks) = pagerank(tcsr, tol);

    if (rank == MASTER) {
        end_time = hrc::now();
        duration = end_time-start_time;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << iterations << " iterations / " << duration.count() << " s]" << std::endl;
        std::cout << "        (MASTER) Work time: " << work_time << " s" << std::endl;
        std::cout << "        (MASTER) Netw time: " << netw_time << " s" << std::endl;
    }
    ////////////////////////////////////////////////////////////////////////////

    // write PageRanks to file
    if (rank == MASTER) {
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
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
