#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph)
{
    // initialization
    const uint_fast32_t n = graph.num_nodes;

    // build the adjacency matrix
    arma::sp_fmat A(n, n);
    for (uint_fast32_t from_node = 0; from_node < graph.num_nodes; ++from_node) {
        const auto outdegree = graph.out_edges.count(from_node) > 0 ? graph.out_edges.at(from_node).size() : 0;
        if (outdegree > 0) {
            for (uint_fast32_t to_node: graph.out_edges.at(from_node)) {
                A(from_node, to_node) = 1.0f/outdegree;
            }
        }
    }
    const arma::sp_fmat At = A.t();

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
        for (auto node: graph.dangling_nodes) {
            sum += p[node];
        }
        dangling.fill((1.0f/n)*sum);

        p_new = (1-d)/n * ones + d * (At*p + dangling);
    }
    while (arma::norm(p_new-p, 1) >= 1E-6f);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < p.size(); ++node) {
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
