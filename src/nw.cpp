#include <cstdint>
#include <vector>

#include "nw.hpp"

#include "utils.hpp"

#include "armadillo"
#include "mpi.h"


int probe(unsigned source, unsigned tag)
{
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
    int count;
    MPI_Get_count(&status, MPI_INT, &count);
    return count;
}


void send_uns(unsigned uns, unsigned destination, unsigned tag)
{
    MPI_Send(&uns, 1, MPI_UNSIGNED, destination, tag, MPI_COMM_WORLD);
}

unsigned recv_uns(unsigned source, unsigned tag)
{
    unsigned uns;
    MPI_Recv(&uns, 1, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return uns;
}


void send_vec(const arma::fvec& vec, unsigned destination, unsigned tag)
{
    MPI_Send(vec.memptr(), vec.size(), MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
}

arma::fvec recv_vec(unsigned source, unsigned tag)
{
    arma::fvec vec(probe(source, tag));
    MPI_Recv(vec.memptr(), vec.size(), MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return vec;
}


void send_mat(const arma::fmat& mat, unsigned destination, unsigned tag)
{
    send_uns(mat.n_rows, destination, tag);
    MPI_Send(mat.memptr(), mat.size(), MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
}

arma::fmat recv_mat(unsigned source, unsigned tag)
{
    const unsigned n_rows = recv_uns(source, tag);

    arma::fmat mat(probe(source, tag), 1);
    MPI_Recv(mat.memptr(), mat.size(), MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    mat.reshape(n_rows, mat.size()/n_rows);
    return mat;
}


void send_csr_vec_f(const std::vector<float>& vec, unsigned destination, unsigned tag)
{
    MPI_Send(vec.data(), vec.size(), MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
}

std::vector<float> recv_csr_vec_f(unsigned source, unsigned tag)
{
    std::vector<float> vec(probe(source, tag));
    MPI_Recv(vec.data(), vec.size(), MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return vec;
}

void send_csr_vec_i(const std::vector<uint_fast32_t>& vec, unsigned destination, unsigned tag)
{
    MPI_Send(vec.data(), vec.size(), MPI_UNSIGNED, destination, tag, MPI_COMM_WORLD);
}

std::vector<uint_fast32_t> recv_csr_vec_i(unsigned source, unsigned tag)
{
    std::vector<uint_fast32_t> vec(probe(source, tag));
    MPI_Recv(vec.data(), vec.size(), MPI_UNSIGNED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return vec;
}

void send_csr(const CSR& csr, unsigned destination, unsigned tag)
{
    send_uns(csr.num_rows, destination, tag);
    send_uns(csr.num_cols, destination, tag);
    send_csr_vec_f(csr.a, destination, tag);
    send_csr_vec_i(csr.ia, destination, tag);
    send_csr_vec_i(csr.ja, destination, tag);
}

CSR recv_csr(unsigned source, unsigned tag)
{
    CSR csr;
    csr.num_rows = recv_uns(source, tag);
    csr.num_cols = recv_uns(source, tag);
    csr.a = recv_csr_vec_f(source, tag);
    csr.ia = recv_csr_vec_i(source, tag);
    csr.ja = recv_csr_vec_i(source, tag);
    return csr;
}
