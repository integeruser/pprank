#include <tuple>
#include <vector>

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
                std::vector<uint_fast32_t> displacements, sizes;
                std::vector<TCSR> tcsrs;
                std::tie(displacements, sizes, tcsrs) = tcsr.split(1);

                REQUIRE(displacements.size() == 1);
                REQUIRE(sizes.size() == 1);
                REQUIRE(tcsrs.size() == 1);

                REQUIRE(tcsrs[0].num_rows == 3);
                REQUIRE(tcsrs[0].num_cols == 3);
                REQUIRE(tcsrs[0].a == ((const std::vector<pprank_t>) {
                    1.0, 1.0
                }));
                REQUIRE(tcsrs[0].ia == ((const std::vector<uint_fast32_t>) {
                    0, 1, 2, 2
                }));
                REQUIRE(tcsrs[0].ja == ((const std::vector<uint_fast32_t>) {
                    1, 2
                }));
            }

            SECTION( "2" ) {
                std::vector<uint_fast32_t> displacements, sizes;
                std::vector<TCSR> tcsrs;
                std::tie(displacements, sizes, tcsrs) = tcsr.split(2);

                REQUIRE(displacements.size() == 2);
                REQUIRE(sizes.size() == 2);
                REQUIRE(tcsrs.size() == 2);

                REQUIRE(tcsrs[0].num_rows == 2);
                REQUIRE(tcsrs[0].num_cols == 3);
                REQUIRE(tcsrs[0].a == ((const std::vector<pprank_t>) {
                    1.0, 1.0
                }));
                REQUIRE(tcsrs[0].ia == ((const std::vector<uint_fast32_t>) {
                    0, 1, 2
                }));
                REQUIRE(tcsrs[0].ja == ((const std::vector<uint_fast32_t>) {
                    1, 2
                }));

                REQUIRE(tcsrs[1].num_rows == 1);
                REQUIRE(tcsrs[1].num_cols == 3);
                REQUIRE(tcsrs[1].a == ((const std::vector<pprank_t>) {
                }));
                REQUIRE(tcsrs[1].ia == ((const std::vector<uint_fast32_t>) {
                    0, 0
                }));
                REQUIRE(tcsrs[1].ja == ((const std::vector<uint_fast32_t>) {
                }));
            }

            SECTION( "3" ) {
                std::vector<uint_fast32_t> displacements, sizes;
                std::vector<TCSR> tcsrs;
                std::tie(displacements, sizes, tcsrs) = tcsr.split(3);

                REQUIRE(displacements.size() == 3);
                REQUIRE(sizes.size() == 3);
                REQUIRE(tcsrs.size() == 3);

                REQUIRE(tcsrs[0].num_rows == 1);
                REQUIRE(tcsrs[0].num_cols == 3);
                REQUIRE(tcsrs[0].a == ((const std::vector<pprank_t>) {
                    1.0
                }));
                REQUIRE(tcsrs[0].ia == ((const std::vector<uint_fast32_t>) {
                    0, 1
                }));
                REQUIRE(tcsrs[0].ja == ((const std::vector<uint_fast32_t>) {
                    1
                }));

                REQUIRE(tcsrs[1].num_rows == 1);
                REQUIRE(tcsrs[1].num_cols == 3);
                REQUIRE(tcsrs[1].a == ((const std::vector<pprank_t>) {
                    1.0
                }));
                REQUIRE(tcsrs[1].ia == ((const std::vector<uint_fast32_t>) {
                    0, 1
                }));
                REQUIRE(tcsrs[1].ja == ((const std::vector<uint_fast32_t>) {
                    2
                }));

                REQUIRE(tcsrs[2].num_rows == 1);
                REQUIRE(tcsrs[2].num_cols == 3);
                REQUIRE(tcsrs[2].a == ((const std::vector<pprank_t>) {
                }));
                REQUIRE(tcsrs[2].ia == ((const std::vector<uint_fast32_t>) {
                    0, 0
                }));
                REQUIRE(tcsrs[2].ja == ((const std::vector<uint_fast32_t>) {
                }));
            }
        }
    }
}
