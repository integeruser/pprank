#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::map<uint_fast32_t, float> rank(const Graph& graph) {
    // initialization
    const auto n = graph.edges.size();

    const auto d = 0.85f;

    arma::vec p(n), p_prev;
    p.fill(1.0f/n);

    arma::mat A(n, n);
    for (size_t i = 0; i < n; ++i) {
        const auto outdegree = graph.edges.at(i).size();
        if (outdegree == 0) {
            // dangling node
            for (size_t j = 0; j < n; ++j) {
                A(i, j) = 1.0f/n;
            }
        }
        else {
            for (const auto j: graph.edges.at(i)) {
                A(i, j) = 1.0f/outdegree;
            }
        }
    }

    const auto At = A.t();

    const auto ones = arma::ones<arma::vec>(n);

    // PageRank computation
    size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;
        p = (1-d)/n * ones + d * (At*p);
    } while (arma::norm(p-p_prev) >= 1E-6f);
    std::cout << "Ended in " << iterations << " iterations" << std::endl;

    std::map<uint_fast32_t, float> ranks;
    for (size_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
    }
    return ranks;
}


int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    const auto filename = argv[1];
    const auto graph = Graph(filename);
    std::cout << "Edges: " << graph.edges << std::endl;

    const auto ranks = rank(graph);
    std::cout << "Ranks: " << ranks << std::endl;

    return EXIT_SUCCESS;
}