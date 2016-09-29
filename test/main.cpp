#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"


TEST_CASE( "sparse matrix-vector construction" ) {
    SECTION( "from matrix" ) {
        SECTION( "1x3" ) {
            arma::mat A(1, 3);
            A(0, 0) = 3.14;
            A(0, 1) = 0;
            A(0, 2) = 1337;

            const auto csr = CSR(A);

            REQUIRE(csr.n_rows == A.n_rows);
            REQUIRE(csr.n_cols == A.n_cols);
            REQUIRE(csr.a == ((const std::vector<float>){3.14f, 1337.0f}));
            REQUIRE(csr.ia == ((const std::vector<uint_fast32_t>){0, 2}));
            REQUIRE(csr.ja == ((const std::vector<uint_fast32_t>){0, 2}));
        }

        SECTION( "3x3" ) {
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

            const auto csr = CSR(A);

            REQUIRE(csr.n_rows == A.n_rows);
            REQUIRE(csr.n_cols == A.n_cols);
            REQUIRE(csr.a == ((const std::vector<float>){1.0f, 1337.0f, 1.0f, 42.0f}));
            REQUIRE(csr.ia == ((const std::vector<uint_fast32_t>){0, 2, 3, 4}));
            REQUIRE(csr.ja == ((const std::vector<uint_fast32_t>){0, 2, 2, 1}));
        }
    }

    SECTION( "from graph" ) {
        SECTION( "simple.txt" ) {
            const auto graph = Graph("inputs/simple.txt");

            const auto csr = CSR(graph);

            REQUIRE(csr.n_rows == graph.nodes.size());
            REQUIRE(csr.n_cols == graph.nodes.size());
            REQUIRE(csr.a == ((const std::vector<float>){1.0f, 1.0f}));
            REQUIRE(csr.ia == ((const std::vector<uint_fast32_t>){0, 1, 2, 2}));
            REQUIRE(csr.ja == ((const std::vector<uint_fast32_t>){1, 2}));
        }
    }
}

TEST_CASE( "sparse matrix-vector multiplication" ) {
    SECTION( "1x1 matrix times 1x1 vector" ) {
        arma::mat A(1, 1);
        A(0, 0) = 1337;

        arma::vec b(1);
        b(0) = -42.42;

        const auto x1 = arma::vec(A*b);
        const auto x2 = CSR(A)*b;
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
        const auto x2 = CSR(A)*b;
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
        const auto x2 = CSR(A)*b;
        REQUIRE(arma::approx_equal(x1, x2, "absdiff", 10e-3));
    }
}

TEST_CASE( "sparse matrix splitting" ) {
    SECTION( "split 1x1 matrix into 1 submatrix" ) {
        arma::mat A(1, 1);
        A(0, 0) = 1337;

        const auto csr = CSR(A);

        const auto split = csr.split(1);

        REQUIRE(split.size() == 1);

        REQUIRE(split[0].a == ((const std::vector<float>){1337.0f}));
        REQUIRE(split[0].ia == ((const std::vector<uint_fast32_t>){0, 1}));
        REQUIRE(split[0].ja == ((const std::vector<uint_fast32_t>){0}));
    }

    SECTION( "split 2x2 matrix into 2 submatrices" ) {
        arma::mat A(2, 4);
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

        REQUIRE(split[0].a == ((const std::vector<float>){1.0f, 1337.0f}));
        REQUIRE(split[0].ia == ((const std::vector<uint_fast32_t>){0, 2}));
        REQUIRE(split[0].ja == ((const std::vector<uint_fast32_t>){0, 2}));

        REQUIRE(split[1].a == ((const std::vector<float>){42.42f, -31337.0f, 3.14f}));
        REQUIRE(split[1].ia == ((const std::vector<uint_fast32_t>){0, 3}));
        REQUIRE(split[1].ja == ((const std::vector<uint_fast32_t>){0, 1, 2}));
    }
}
