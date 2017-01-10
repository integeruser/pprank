#ifndef NW_HPP
#define NW_HPP

#include "armadillo"


void send_uns(unsigned val, unsigned destination, unsigned tag);

unsigned recv_uns(unsigned source, unsigned tag);


void send_vec(const arma::vec& vec, unsigned destination, unsigned tag);

arma::vec recv_vec(unsigned source, unsigned tag);


void send_mat(const arma::mat& mat, unsigned destination, unsigned tag);

arma::mat recv_mat(unsigned source, unsigned tag);


#endif
