#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "mpi.h"
#include "prettyprint.hpp"

#define MASTER 0

int rank;
int num_processes;


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph)
{
    // initialization
    const uint_fast32_t n = graph.num_nodes;

    const CSC A = CSC(graph);
    const CSR At = transpose(A);

    arma::fvec p, p_new(n), dangling(n);
    p_new.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const float d = 0.85f;

    // partition At in blocks of rows
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
    do {
        ++iterations;

        p = p_new;

        float sum = 0.0f;
        for (uint_fast32_t node: graph.dangling_nodes) {
            sum += p[node];
        }
        sum *= 1.0f/n;
        dangling.fill(sum);

        // each process computes the matrix-vector product only for its block of rows
        const CSR At_block = blocks[rank].second;
        const arma::fvec prod_block = At_block * p;

        // gather the results of the matrix-vector products
        arma::fvec prod(n);
        MPI_Allgatherv(prod_block.memptr(), blocks_num_rows[rank], MPI_FLOAT,
                       prod.memptr(), blocks_num_rows.data(), blocks_displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);

        p_new = (1-d)/n * ones + d * (prod + dangling);
    }
    while (arma::norm(p_new-p, 1) >= 1E-6f);

    const double finish_time = MPI_Wtime();

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < n; ++node) {
        ranks[node] = p[node];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (argc != 2) {
        if (rank == MASTER) {
            std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        }
        return EXIT_FAILURE;
    }

    const auto filename = argv[1];
    const auto graph = Graph(filename);
    if (rank == MASTER) {
        std::cout << "Nodes: " << graph.num_nodes << std::endl;
    }

    const auto results = pagerank(graph);
    if (rank == MASTER) {
        const auto iterations = results.first;
        const auto ranks = results.second;
        std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
