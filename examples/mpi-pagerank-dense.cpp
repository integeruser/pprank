#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "ds.hpp"
#include "nw.hpp"

#include "armadillo"
#include "mpi.h"
#include "prettyprint.hpp"


std::vector<std::pair<unsigned, const arma::mat>> split(const auto& mat, unsigned slave_count)
{
    std::vector<std::pair<unsigned, const arma::mat>> chunks;
    unsigned num_processed_rows = 0;
    for (unsigned i = 0; i < slave_count; ++i) {
        const unsigned offset = num_processed_rows;
        const unsigned n_rows = mat.n_rows/slave_count + (i < mat.n_rows%slave_count);
        const arma::mat& chunk = mat.rows(offset, offset+(n_rows-1));
        chunks.emplace_back(offset, chunk);
        num_processed_rows += n_rows;
    }
    return chunks;
}

std::pair<std::size_t, std::map<NodeIndex, float>> pagerank(const Graph& graph, unsigned slave_count, unsigned tag)
{
    // initialization
    const std::size_t n = graph.edges.size();
    const auto d = 0.85f;
    const arma::vec ones(n, arma::fill::ones);

    arma::vec p(n), p_prev;
    p.fill(1.0f/n);

    const auto A = graph.to_dense();
    const arma::mat At = A.t();
    assert(slave_count <= A.n_rows);

    // split At in submatrices and send them to slaves
    const auto chunks = split(At, slave_count);
    for (unsigned i = 0; i < slave_count; ++i) {
        const auto& chunk = chunks.at(i).second;
        const auto slave = i+1;
        send_mat(chunk, slave, tag);
    }

    // ranks computation
    std::size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;

        // tell the slaves that there is still work to do
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto slave = i+1;
            send_uns(0, slave, tag);
        }

        // send p to each slave
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto slave = i+1;
            send_vec(p, slave, tag);
        }

        // receive back the matrix-vector products
        auto prod = arma::vec(A.n_rows);
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto slave = i+1;
            const auto mat = recv_mat(slave, tag);
            const auto offset = chunks.at(i).first;
            for (std::size_t j = 0; j < mat.n_rows; ++j) {
                prod(offset+j) = mat(j, 0);
            }
        }

        p = (1-d)/n * ones + d * prod;
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // shut down the slaves
    for (size_t slave = 1; slave <= slave_count; ++slave) {
        send_uns(0x1337, slave, tag);
    }

    // map each node to its rank
    std::map<NodeIndex, float> ranks;
    for (std::size_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
    }
    return std::make_pair(iterations, ranks);
}

void master_do(char const *argv[], unsigned slave_count, unsigned tag)
{
    const auto filename = argv[1];
    const auto graph = Graph(filename);
    std::cout << "Edges: " << graph.edges << std::endl;

    const auto results = pagerank(graph, slave_count, tag);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;
}


void slave_do(int rank, unsigned master, unsigned tag)
{
    // receive from the master the submatrix to process
    const auto At = recv_mat(master, tag);

    while (true) {
        // check if there is no more work to be done
        const auto message = recv_uns(master, tag);
        if (message == 0x1337) {
            break;
        }

        // receive from the master the vector to process
        const auto p = recv_vec(master, tag);

        // compute the new probabilities
        const auto prod = At*p;
        send_mat(prod, master, tag);
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

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    assert(process_count >= 2);

    const unsigned slave_count = process_count-1;

    const unsigned master = 0;
    const unsigned tag = 0;

    if (rank == 0) {
        master_do(argv, slave_count, tag);
    }
    else {
        slave_do(rank, master, tag);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
