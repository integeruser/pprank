#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "nw.hpp"
#include "utils.hpp"

#include "armadillo"
#include "mpi.h"
#include "prettyprint.hpp"

#define MASTER 0
#define TAG 0


std::pair<size_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph, unsigned num_slaves)
{
    // initialization
    uint_fast32_t n = graph.num_nodes;
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, MASTER, MPI_COMM_WORLD);

    const CSC A = CSC(graph);
    const CSR At = transpose(A);
    assert(num_slaves <= At.num_rows);

    arma::fvec p(n), p_prev;
    p.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const float d = 0.85f;

    // partition At in blocks of rows and send them to the slaves
    const auto blocks = At.split(num_slaves);
    for (unsigned slave = 1; slave <= num_slaves; ++slave) {
        const CSR block = blocks.at(slave-1).second;
        send_csr(block, slave, TAG);
    }

    // ranks computation
    size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;

        // tell the slaves that there is still work to do
        unsigned msg = 0x42;
        MPI_Bcast(&msg, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);
        // broadcast p to the slaves
        MPI_Bcast(p.memptr(), n, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

        // receive back from the slaves the matrix-vector products
        arma::fvec prod(n);
        for (unsigned slave = 1; slave <= num_slaves; ++slave) {
            const uint_fast32_t offset = blocks.at(slave-1).first;

            const arma::fvec vec = recv_vec(slave, TAG);
            for (uint_fast32_t i = 0; i < vec.size(); ++i) {
                prod(offset+i) = vec(i);
            }
        }

        p = (1-d)/n * ones + d * (prod);
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // shut down the slaves
    unsigned msg = 0x1337;
    MPI_Bcast(&msg, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
    }
    return std::make_pair(iterations, ranks);
}

void master_do(char const *argv[], unsigned num_slaves)
{
    const auto filename = argv[1];
    const auto graph = Graph(filename);
    std::cout << "Nodes: " << graph.num_nodes << std::endl;

    const auto results = pagerank(graph, num_slaves);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;
}


void slave_do(int rank)
{
    uint_fast32_t n;
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, MASTER, MPI_COMM_WORLD);

    // receive the matrix to process
    const CSR csr = recv_csr(MASTER, TAG);

    while (true) {
        // check if there is still work to do
        unsigned msg;
        MPI_Bcast(&msg, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);
        // const unsigned msg = recv_uns(MASTER, TAG);
        if (msg == 0x1337) {
            break;
        }

        // receive the vector to process
        arma::fvec vec(n);
        MPI_Bcast(vec.memptr(), n, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        // compute and send back the matrix-vector product
        send_vec(csr*vec, MASTER, TAG);
    }
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    const unsigned num_slaves = num_processes-1;

    if (rank == 0) {
        master_do(argv, num_slaves);
    }
    else {
        slave_do(rank);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
