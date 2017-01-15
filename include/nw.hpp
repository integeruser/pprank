#ifndef NW_HPP
#define NW_HPP

#include "utils.hpp"

#include "armadillo"


void send_uns(unsigned val, unsigned destination, unsigned tag);

unsigned recv_uns(unsigned source, unsigned tag);


void send_vec(const arma::fvec& vec, unsigned destination, unsigned tag);

arma::fvec recv_vec(unsigned source, unsigned tag);


void send_mat(const arma::fmat& mat, unsigned destination, unsigned tag);

arma::fmat recv_mat(unsigned source, unsigned tag);


void send_csr(const CSR& csr, unsigned destination, unsigned tag);

CSR recv_csr(unsigned source, unsigned tag);


#endif
