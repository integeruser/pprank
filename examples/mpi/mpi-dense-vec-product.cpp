#include <cassert>
#include <vector>

#include "armadillo"
#include "mpi.h"

#include "nw.hpp"


int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    assert(process_count >= 2);

    const unsigned master = 0;
    const unsigned tag = 0;

    if (rank == 0) {
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

        arma::vec vec(4);
        vec(0) = -1.0;
        vec(1) = 1.0;
        vec(2) = 21.5;
        vec(3) = 42.0;

        assert(mat.n_cols == vec.size());

        const unsigned slave_count = process_count-1;
        assert(slave_count <= mat.n_rows);

        // send to each slave the vector to process
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto slave = i+1;
            send_vec(vec, slave, tag);
        }

        // count how many rows send to each slave process
        std::vector<unsigned> row_count(slave_count);
        for (size_t i = 0; i < mat.n_rows; ++i) {
            ++row_count[i%slave_count];
        }
        std::vector<unsigned> offsets(slave_count);
        unsigned offset = 0;
        for (unsigned i = 0; i < slave_count; ++i) {
            offsets[i] = offset;
            offset += row_count[i];
        }

        // send to each slave the number of rows of the matrix to process
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto val = row_count[i];
            const auto slave = i+1;
            send_uns(val, slave, tag);
        }

        // split the matrix into (roughly) evenly sized chunks,
        // and send each chunk to a different process
        unsigned row_index = 0;
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto& submat = mat.rows(row_index, row_index+(row_count[i]-1));
            row_index += row_count[i];
            const auto slave = i+1;
            send_mat(submat, slave, tag);
        }
        assert(row_index == mat.n_rows);

        // receive the dot products
        auto res = arma::vec(mat.n_rows);
        for (unsigned i = 0; i < slave_count; ++i) {
            const auto slave = i+1;
            const auto n_rows = row_count[i];
            const auto& mat = recv_mat(slave, tag, n_rows);
            for (size_t j = 0; j < mat.n_rows; ++j) {
                res(offsets[i]+j) = mat(j, 0);
            }
        }

        std::cout << "Result: " << std::endl;
        std::cout << res << std::endl;
        assert(arma::approx_equal(res, arma::vec{1699.5f, 491.0f, -5783.5f, -53.5f}, "absdiff", 10e-5));
    }
    else {
        // receive from the master the vector to process
        const auto& vec = recv_vec(master, tag);

        // receive from the master the number of rows of the matrix to process
        const auto n_rows = recv_uns(master, tag);

        // receive from the master the matrix to process
        const auto& mat = recv_mat(master, tag, n_rows);

        // compute the dot product and send the result to the master
        send_mat(mat*vec, master, tag);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
