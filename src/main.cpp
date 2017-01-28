#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "mpi.h"
#include "prettyprint.hpp"

#define MASTER 0

int rank;
int num_processes;


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph, const float tol)
{
    // initialization
    const uint_fast32_t N = graph.num_nodes;

    // build the adjacency matrix
    if (rank == MASTER) { std::cout << "[*] Building adjacency matrix..." << std::endl; }
    const CSC A = CSC(graph);
    const CSR At = transpose(A);

    arma::fvec p, p_new(N);
    p_new.fill(1.0f/N);

    // find dangling nodes
    if (rank == MASTER) { std::cout << "[*] Finding dangling nodes..." << std::endl; }
    arma::fvec dangling(N);
    const arma::uvec dangling_nodes =
        arma::conv_to<arma::uvec>::from(
            std::vector<uint_fast32_t>(graph.dangling_nodes.cbegin(), graph.dangling_nodes.cend()));

    const arma::fvec ones(N, arma::fill::ones);
    const float d = 0.85f;

    // partition At in blocks of rows
    if (rank == MASTER) { std::cout << "[*] Splitting matrix in blocks..." << std::endl; }
    const auto blocks = At.split(num_processes);

    std::vector<int> blocks_displacements, blocks_num_rows;
    for (const auto block: blocks) {
        blocks_displacements.push_back(block.first);
        blocks_num_rows.push_back(block.second.num_rows);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    // ranks computation
    uint_fast32_t iterations = 0;
    if (rank == MASTER) { std::cout << "[*] Computing PageRank (tol=" << tol << ")..." << std::endl; }
    do {
        ++iterations;

        p = p_new;

        dangling.fill(1.0f/N * arma::sum(p(dangling_nodes)));

        // each process computes the matrix-vector product only for its block of rows
        const CSR At_block = blocks[rank].second;
        const arma::fvec prod_block = At_block * p;

        // gather the results of the matrix-vector products
        arma::fvec prod(N);
        MPI_Allgatherv(prod_block.memptr(), blocks_num_rows[rank], MPI_FLOAT,
                       prod.memptr(), blocks_num_rows.data(), blocks_displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);

        p_new = (1-d)/N * ones + d * (prod + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);

    const double finish_time = MPI_Wtime();

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < N; ++node) {
        ranks[node] = p[node];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (!(argc == 2 || argc == 3)) {
        if (rank == MASTER) { std::cerr << "Usage: pprank file [tol]" << std::endl; }
        return EXIT_FAILURE;
    }

    char* file = argv[1];
    float tol = 1e-6f;
    if (argc == 3) {
        tol = std::atof(argv[2]);
    }

    if (rank == MASTER) { std::cout << "[*] Building graph..." << std::endl; }
    const auto graph = Graph(file);
    if (rank == MASTER) { std::cout << "        Nodes: " << graph.num_nodes << std::endl; }

    const auto results = pagerank(graph, tol);
    const auto iterations = results.first;
    const auto ranks = results.second;
    if (rank == MASTER) { std::cout << "[*] Ranks: " << ranks << " in " << iterations << " iterations " << std::endl; }

    if (rank == MASTER) {
        std::ofstream outfile;
        outfile.open("ranks-" + std::to_string(graph.num_nodes) + ".txt");
        outfile << ranks;
        outfile.close();
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}