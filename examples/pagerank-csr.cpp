#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<size_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph)
{
    // initialization
    const uint_fast32_t n = graph.num_nodes;

    const CSC A = CSC(graph);
    const CSR At = transpose(A);

    arma::fvec p, p_new(n), dangling(n);
    p_new.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    do {
        ++iterations;

        p = p_new;

        float sum = 0.0f;
        for (uint_fast32_t node: graph.dangling_nodes) {
            sum += p[node];
        }
        sum *= 1.0f/n;
        dangling.fill(sum);

        p_new = (1-d)/n * ones + d * (At*p + dangling);
    }
    while (arma::norm(p_new-p, 1) >= 1E-6f);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < n; ++node) {
        ranks[node] = p[node];
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
    std::cout << "Nodes: " << graph.num_nodes << std::endl;

    const auto results = pagerank(graph);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;

    return EXIT_SUCCESS;
}
