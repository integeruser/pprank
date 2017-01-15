#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


arma::sp_fmat to_adjacency_mat(const Graph& graph)
{
    arma::sp_fmat adjmat(graph.num_nodes, graph.num_nodes);
    for (uint_fast32_t i = 0; i < graph.num_nodes; ++i) {
        const auto outdegree = graph.out_edges.count(i) > 0 ? graph.out_edges.at(i).size() : 0;
        if (outdegree == 0) {
            // dangling node
            for (uint_fast32_t j = 0; j < graph.num_nodes; ++j) {
                adjmat(i, j) = 1.0f/graph.num_nodes;
            }
        }
        else {
            for (uint_fast32_t j: graph.out_edges.at(i)) {
                adjmat(i, j) = 1.0f/outdegree;
            }
        }
    }
    return adjmat;
}

std::pair<size_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph)
{
    // initialization
    const size_t n = graph.num_nodes;

    const auto A = to_adjacency_mat(graph);
    const auto At = A.t();

    arma::fvec p(n), p_prev;
    p.fill(1.0f/n);

    const arma::fvec ones(n, arma::fill::ones);
    const auto d = 0.85f;

    // ranks computation
    size_t iterations = 0;
    do {
        iterations += 1;

        p_prev = p;

        p = (1-d)/n * ones + d * (At*p);
    }
    while (arma::norm(p-p_prev) >= 1E-6f);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t i = 0; i < p.size(); ++i) {
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
    std::cout << "Nodes: " << graph.num_nodes << std::endl;

    const auto results = pagerank(graph);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "Ranks: " << ranks << " in " << iterations << " iterations " << std::endl;

    return EXIT_SUCCESS;
}
