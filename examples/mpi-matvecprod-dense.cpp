#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "nw.hpp"

#include "armadillo"
#include "mpi.h"


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

void master_do(unsigned slave_count, unsigned tag)
{
    arma::mat mat(4, 4);
    mat(0, 0) = 1.0;
    mat(0, 1) = 1337.0;
    mat(0, 2) = 13.0;
    mat(0, 3) = 2.0;
    mat(1, 0) = 14.0;
    mat(1, 1) = 1.0;
    mat(1, 2) = 0.0;
    mat(1, 3) = 12.0;
    mat(2, 0) = 65.0;
    mat(2, 1) = 34.0;
    mat(2, 2) = 45.0;
    mat(2, 3) = -160.0;
    mat(3, 0) = 76.0;
    mat(3, 1) = 23.0;
    mat(3, 2) = -1.0;
    mat(3, 3) = 0.5;
    assert(slave_count <= mat.n_rows);

    arma::vec vec(4);
    vec(0) = -1.0;
    vec(1) = 1.0;
    vec(2) = 21.5;
    vec(3) = 42.0;
    assert(mat.n_cols == vec.size());

    // split the matrix in submatrices and send them to slaves
    const auto chunks = split(mat, slave_count);
    for (unsigned i = 0; i < slave_count; ++i) {
        const auto& chunk = chunks.at(i).second;
        const auto slave = i+1;
        send_mat(chunk, slave, tag);
    }

    // send the vector to each slave
    for (unsigned i = 0; i < slave_count; ++i) {
        const auto slave = i+1;
        send_vec(vec, slave, tag);
    }

    // receive back the matrix-vector products
    auto prod = arma::vec(mat.n_rows);
    for (unsigned i = 0; i < slave_count; ++i) {
        const auto slave = i+1;
        const auto mat = recv_mat(slave, tag);
        const auto offset = chunks.at(i).first;
        for (std::size_t j = 0; j < mat.n_rows; ++j) {
            prod(offset+j) = mat(j, 0);
        }
    }

    std::stringstream ss;
    ss << "Result: " << std::endl << prod;
    std::cout << ss.str() << std::endl;
    assert(arma::approx_equal(prod, arma::vec{1699.5f, 491.0f, -5783.5f, -53.5f}, "absdiff", 10e-5));
}


void slave_do(int rank, unsigned master, unsigned tag)
{
    std::stringstream ss;
    ss << "----------" << std::endl;
    ss << "Hello from slave " << rank << "!" << std::endl;

    // receive from the master the submatrix to process
    const auto mat = recv_mat(master, tag);
    ss << "I have received" << std::endl << mat;

    // receive from the master the vector to process
    const auto vec = recv_vec(master, tag);
    ss << "and" << std::endl << vec;

    // compute the multiplication between the two and send the result back to the master
    const auto prod = mat*vec;
    send_mat(prod, master, tag);
    ss << "to compute" << std::endl << prod;

    ss << "----------";
    std::cout << ss.str() << std::endl;
}


int main(int argc, char const *argv[])
{
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
        master_do(slave_count, tag);
    }
    else {
        slave_do(rank, master, tag);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
