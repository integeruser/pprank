#include "mpi.h"

#include "nw.hpp"


void send_uns(unsigned val, unsigned destination, unsigned tag) {
    MPI_Send(&val, 1, MPI_UNSIGNED, destination, tag, MPI_COMM_WORLD);
}

unsigned recv_uns(unsigned source, unsigned tag) {
    unsigned val;
    MPI_Recv(&val, 1, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return val;
}


void send_vec(const arma::vec& vec, unsigned destination, unsigned tag) {
    MPI_Send(vec.cbegin(), vec.size(), MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
}

arma::vec recv_vec(unsigned source, unsigned tag) {
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

    int count = 0;
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    arma::vec vec(count);
    MPI_Recv(vec.begin(), count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return vec;
}


void send_mat(const arma::mat& mat, unsigned destination, unsigned tag) {
    MPI_Send(mat.cbegin(), mat.size(), MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
}

arma::mat recv_mat(unsigned source, unsigned tag, unsigned n_rows) {
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

    int count = 0;
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    arma::mat mat(count, 1);
    MPI_Recv(mat.begin(), count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mat.reshape(n_rows, mat.size()/n_rows);
    return mat;
}
