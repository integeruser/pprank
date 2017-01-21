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


std::pair<size_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph)
{
    // initialization
    const uint_fast32_t n = graph.num_nodes;

    const CSC A = CSC(graph);
    const CSR At = transpose(A);

    arma::fvec p(n), p_prev;
    p.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const float d = 0.85f;

    // partition At in blocks of rows
    const auto blocks = At.split(num_processes);

    std::vector<int> blocks_displacements, blocks_num_rows;
    for (const auto block: blocks) {
        blocks_displacements.push_back(block.first);
        blocks_num_rows.push_back(block.second.num_rows);
    }

    // ranks computation
    uint_fast32_t iterations = 0;
    do {
        ++iterations;

        p_prev = p;

        // each process computes the matrix-vector product only for its block of rows
        const CSR At_block = blocks[rank].second;
        const arma::fvec prod_block = At_block * p;

        // gather the results of the matrix-vector products
        arma::fvec prod(n);
        MPI_Allgatherv(prod_block.memptr(), blocks_num_rows[rank], MPI_FLOAT,
                    prod.memptr(), blocks_num_rows.data(), blocks_displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);

        p = (1-d)/n * ones + d * (prod);
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
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
