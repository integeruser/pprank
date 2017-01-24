#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "utils.hpp"

#include "armadillo"


TEST_CASE( "sparse matrix-vector construction" )
{
    SECTION( "from graph" ) {
        SECTION( "toy.txt" ) {
            const auto graph = Graph("inputs/toy.txt");

            const auto csc = CSC(graph);

            REQUIRE(csc.num_rows == graph.num_nodes);
            REQUIRE(csc.num_cols == graph.num_nodes);
            REQUIRE(csc.a == ((const std::vector<float>) {
                1.0f, 1.0f
            }));
            REQUIRE(csc.ia == ((const std::vector<uint_fast32_t>) {
                0, 0, 1, 2
            }));
            REQUIRE(csc.ja == ((const std::vector<uint_fast32_t>) {
                0, 1
            }));
        }
    }
}

TEST_CASE( "sparse matrix-vector multiplication" )
{
    SECTION( "1x1 matrix times 1x1 vector" ) {
        arma::fmat A(1, 1);
        A(0, 0) = 1337;

        arma::fvec b(1);
        b(0) = -42.42;

        const auto x1 = arma::fvec(A*b);
        const auto x2 = CSR(A)*b;
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-5));
    }

    SECTION( "3x3 matrix times 3x1 vector" ) {
        arma::fmat A(3, 3);
        A(0, 0) = 1;
        A(0, 1) = 0;
        A(0, 2) = 1337;
        A(1, 0) = 0;
        A(1, 1) = 0;
        A(1, 2) = 1;
        A(2, 0) = 0;
        A(2, 1) = 42;
        A(2, 2) = 0;

        arma::fvec b(3);
        b(0) = 1337;
        b(1) = 0;
        b(2) = -42.42;

        const auto x1 = arma::fvec(A*b);
        const auto x2 = CSR(A)*b;
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-5));
    }

    SECTION( "2x4 matrix times 4x1 vector" ) {
        arma::fmat A(2, 4);
        A(0, 0) = 1;
        A(0, 1) = 0;
        A(0, 2) = 1337;
        A(0, 3) = 0;
        A(1, 0) = 42.42;
        A(1, 1) = -31337;
        A(1, 2) = 3.14;
        A(1, 3) = 0;

        arma::fvec b(4);
        b(0) = 1337;
        b(1) = 0;
        b(2) = -42.42;
        b(3) = 3.14;

        const auto x1 = arma::fvec(A*b);
        const auto x2 = CSR(A)*b;
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-3));
    }
}

TEST_CASE( "sparse matrix splitting" )
{
    SECTION( "split 1x1 matrix into 1 submatrix" ) {
        arma::fmat A(1, 1);
        A(0, 0) = 1337;

        const auto csr = CSR(A);

        const auto split = csr.split(1);

        REQUIRE(split.size() == 1);

        REQUIRE(split[0].second.a == ((const std::vector<float>) {
            1337.0f
        }));
        REQUIRE(split[0].second.ia == ((const std::vector<uint_fast32_t>) {
            0, 1
        }));
        REQUIRE(split[0].second.ja == ((const std::vector<uint_fast32_t>) {
            0
        }));
    }

    SECTION( "split 2x2 matrix into 2 submatrices" ) {
        arma::fmat A(2, 4);
        A(0, 0) = 1;
        A(0, 1) = 0;
        A(0, 2) = 1337;
        A(0, 3) = 0;
        A(1, 0) = 42.42;
        A(1, 1) = -31337;
        A(1, 2) = 3.14;
        A(1, 3) = 0;

        const auto csr = CSR(A);

        const auto split = csr.split(2);

        REQUIRE(split.size() == 2);

        REQUIRE(split[0].second.a == ((const std::vector<float>) {
            1.0f, 1337.0f
        }));
        REQUIRE(split[0].second.ia == ((const std::vector<uint_fast32_t>) {
            0, 2
        }));
        REQUIRE(split[0].second.ja == ((const std::vector<uint_fast32_t>) {
            0, 2
        }));

        REQUIRE(split[1].second.a == ((const std::vector<float>) {
            42.42f, -31337.0f, 3.14f
        }));
        REQUIRE(split[1].second.ia == ((const std::vector<uint_fast32_t>) {
            0, 3
        }));
        REQUIRE(split[1].second.ja == ((const std::vector<uint_fast32_t>) {
            0, 1, 2
        }));
    }
}
