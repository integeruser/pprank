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

    arma::fvec p, p_new(n);
    p_new.fill(1.0f/n);

    // find dangling nodes
    std::cout << "[*] Finding dangling nodes..." << std::endl;
    arma::fvec dangling(n);
    const arma::uvec dangling_nodes =
        arma::conv_to<arma::uvec>::from(
            std::vector<uint_fast32_t>(graph.dangling_nodes.cbegin(), graph.dangling_nodes.cend()));

    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    std::cout << "[*] Starting PageRank..." << std::endl;
    do {
        ++iterations;
        std::cout << "        Iteration #" << iterations << "..." << std::endl;

        p = p_new;

        const float dangling_nodes_contribution = 1.0f/n * arma::sum(p(dangling_nodes));

        // to avoid storing A in memory recompute its rows at every iteration
        p_new.fill(0.0f);
        for (uint_fast32_t from_node = 0; from_node < graph.num_nodes; ++from_node) {
            const uint_fast32_t outdegree = graph.out_edges.count(from_node) > 0 ?
                                            graph.out_edges.at(from_node).size() : 0;
            if (outdegree > 0) {
                for (uint_fast32_t to_node: graph.out_edges.at(from_node)) {
                    p_new[to_node] += (1.0f/outdegree) * p[from_node];
                }
            }
        }
        for (uint_fast32_t node = 0; node < n; ++node) {
            p_new[node] = (1.0f-d)/n + d * (p_new[node]+dangling_nodes_contribution);
        }
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
    std::cout << "[*] Building graph..." << std::endl;
    const auto graph = Graph(filename);
    std::cout << "        Nodes: " << graph.num_nodes << std::endl;

    const auto results = pagerank(graph);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "[*] Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;

    return EXIT_SUCCESS;
}
