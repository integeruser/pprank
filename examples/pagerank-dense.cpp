#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<std::size_t, std::map<NodeIndex, float>> pagerank(const Graph& graph)
{
    // initialization
    const std::size_t n = graph.edges.size();
    const auto d = 0.85f;
    const auto ones = arma::ones<arma::vec>(n);

    arma::vec p(n), p_prev;
    p.fill(1.0f/n);

    arma::mat A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto outdegree = graph.edges.at(i).size();
        if (outdegree == 0) {
            // dangling node
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) = 1.0f/n;
            }
        }
        else {
            for (std::size_t j: graph.edges.at(i)) {
                A(i, j) = 1.0f/outdegree;
            }
        }
    }
    const auto At = A.t();

    // ranks computation
    std::size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;
        p = (1-d)/n * ones + d * (At*p);
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // map each node to its rank
    std::map<NodeIndex, float> ranks;
    for (std::size_t i = 0; i < p.size(); ++i) {
        ranks[i] = p[i];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    const auto filename = argv[1];
    const auto graph = Graph(filename);
    std::cout << "Edges: " << graph.edges << std::endl;

    const auto results = pagerank(graph);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;

    return EXIT_SUCCESS;
}
