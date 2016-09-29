#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"

#include <cassert>


extern arma::vec mul(const CSR&, const arma::vec&, size_t);


TEST_CASE( "sparse matrix-vector multiplication" ) {
    SECTION( "1x1 matrix times 1x1 vector" ) {
        arma::mat A(1, 1);
        A(0, 0) = 1337;

        arma::vec b(1);
        b(0) = -42.42;

        const auto x1 = arma::vec(A*b);
        const auto x2 = mul(CSR(A), b, b.size());
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-5));
    }

    SECTION( "3x3 matrix times 3x1 vector" ) {
        arma::mat A(3, 3);
        A(0, 0) = 1;
        A(0, 1) = 0;
        A(0, 2) = 1337;
        A(1, 0) = 0;
        A(1, 1) = 0;
        A(1, 2) = 1;
        A(2, 0) = 0;
        A(2, 1) = 42;
        A(2, 2) = 0;

        arma::vec b(3);
        b(0) = 1337;
        b(1) = 0;
        b(2) = -42.42;

        const auto x1 = arma::vec(A*b);
        const auto x2 = mul(CSR(A), b, b.size());
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-5));
    }

    SECTION( "2x4 matrix times 4x1 vector" ) {
        arma::mat A(2, 4);
        A(0, 0) = 1;
        A(0, 1) = 0;
        A(0, 2) = 1337;
        A(0, 3) = 0;
        A(1, 0) = 42.42;
        A(1, 1) = -31337;
        A(1, 2) = 3.14;
        A(1, 3) = 0;

        arma::vec b(4);
        b(0) = 1337;
        b(1) = 0;
        b(2) = -42.42;
        b(3) = 3.14;

        const auto x1 = arma::vec(A*b);
        const auto x2 = mul(CSR(A), b, b.size());
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-3));
    }
}
