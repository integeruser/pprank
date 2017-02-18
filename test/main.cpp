#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "utils.hpp"

#include "armadillo"


TEST_CASE( "sparse matrix construction" )
{
    SECTION( "from graph" ) {
        SECTION( "toy.txt" ) {
            const TCSR tcsr = TCSR("inputs/toy-3-2.txt");

            REQUIRE(tcsr.num_rows == 3);
            REQUIRE(tcsr.num_cols == 3);
            REQUIRE(tcsr.a == ((const std::vector<pprank_t>) {
                1.0, 1.0
            }));
            REQUIRE(tcsr.ia == ((const std::vector<uint_fast32_t>) {
                0, 1, 2, 2
            }));
            REQUIRE(tcsr.ja == ((const std::vector<uint_fast32_t>) {
                1, 2
            }));
        }
    }
}

TEST_CASE( "sparse matrix-vector product with the matrix transposed" )
{
    SECTION( "from graph" ) {
        SECTION( "toy.txt" ) {
            const TCSR tcsr = TCSR("inputs/toy-3-2.txt");

            pprank_vec_t vec(3);
            vec(0) = 1337;
            vec(1) = 0;
            vec(2) = -42.42;

            const pprank_vec_t res = tcsr.tdot(vec);
            REQUIRE(arma::approx_equal(res, (pprank_vec_t) {0, 1337, 0}, "absdiff", 10e-5));
        }
    }
}

TEST_CASE( "sparse matrix split" )
{
    SECTION( "from graph" ) {
        SECTION( "toy.txt" ) {
            const TCSR tcsr = TCSR("inputs/toy-3-2.txt");

            SECTION( "1" ) {
                const std::vector<std::pair<uint_fast32_t, TCSR>> split = tcsr.split(1);

                REQUIRE(split.size() == 1);

                REQUIRE(split[0].second.num_rows == 3);
                REQUIRE(split[0].second.num_cols == 3);
                REQUIRE(split[0].second.a == ((const std::vector<pprank_t>) {
                    1.0, 1.0
                }));
                REQUIRE(split[0].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 1, 2, 2
                }));
                REQUIRE(split[0].second.ja == ((const std::vector<uint_fast32_t>) {
                    1, 2
                }));
            }

            SECTION( "2" ) {
                const std::vector<std::pair<uint_fast32_t, TCSR>> split = tcsr.split(2);

                REQUIRE(split.size() == 2);

                REQUIRE(split[0].second.num_rows == 2);
                REQUIRE(split[0].second.num_cols == 3);
                REQUIRE(split[0].second.a == ((const std::vector<pprank_t>) {
                    1.0, 1.0
                }));
                REQUIRE(split[0].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 1, 2
                }));
                REQUIRE(split[0].second.ja == ((const std::vector<uint_fast32_t>) {
                    1, 2
                }));

                REQUIRE(split[1].second.num_rows == 1);
                REQUIRE(split[1].second.num_cols == 3);
                REQUIRE(split[1].second.a == ((const std::vector<pprank_t>) {
                }));
                REQUIRE(split[1].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 0
                }));
                REQUIRE(split[1].second.ja == ((const std::vector<uint_fast32_t>) {
                }));
            }

            SECTION( "3" ) {
                const std::vector<std::pair<uint_fast32_t, TCSR>> split = tcsr.split(3);

                REQUIRE(split.size() == 3);

                REQUIRE(split[0].second.num_rows == 1);
                REQUIRE(split[0].second.num_cols == 3);
                REQUIRE(split[0].second.a == ((const std::vector<pprank_t>) {
                    1.0
                }));
                REQUIRE(split[0].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 1
                }));
                REQUIRE(split[0].second.ja == ((const std::vector<uint_fast32_t>) {
                    1
                }));

                REQUIRE(split[1].second.num_rows == 1);
                REQUIRE(split[1].second.num_cols == 3);
                REQUIRE(split[1].second.a == ((const std::vector<pprank_t>) {
                    1.0
                }));
                REQUIRE(split[1].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 1
                }));
                REQUIRE(split[1].second.ja == ((const std::vector<uint_fast32_t>) {
                    2
                }));

                REQUIRE(split[2].second.num_rows == 1);
                REQUIRE(split[2].second.num_cols == 3);
                REQUIRE(split[2].second.a == ((const std::vector<pprank_t>) {
                }));
                REQUIRE(split[2].second.ia == ((const std::vector<uint_fast32_t>) {
                    0, 0
                }));
                REQUIRE(split[2].second.ja == ((const std::vector<uint_fast32_t>) {
                }));
            }
        }
    }
}
