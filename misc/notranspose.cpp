#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph, const float tol)
{
    // initialization
    const uint_fast32_t N = graph.num_nodes;

    arma::fvec p, p_new(N);
    p_new.fill(1.0f/N);

    // find dangling nodes
    std::cout << "[*] Finding dangling nodes..." << std::endl;
    arma::fvec dangling(N);
    const arma::uvec dangling_nodes =
        arma::conv_to<arma::uvec>::from(
            std::vector<uint_fast32_t>(graph.dangling_nodes.cbegin(), graph.dangling_nodes.cend()));

    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    std::cout << "[*] Computing PageRank (tol=" << tol << ")..." << std::endl;
    do {
        ++iterations;

        p = p_new;

        const float dangling_nodes_contribution = 1.0f/N * arma::sum(p(dangling_nodes));

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
        for (uint_fast32_t node = 0; node < N; ++node) {
            p_new[node] = (1.0f-d)/N + d * (p_new[node]+dangling_nodes_contribution);
        }
    }
    while (arma::norm(p_new-p, 1) >= tol);

    // map each node to its rank
    std::map<uint_fast32_t, float> ranks;
    for (uint_fast32_t node = 0; node < N; ++node) {
        ranks[node] = p[node];
    }
    return std::make_pair(iterations, ranks);
}


int main(int argc, char const *argv[])
{
    if (!(argc == 2 || argc == 3)) {
        std::cerr << "Usage: pagerank-notranspose file [tol]" << std::endl;
        return EXIT_FAILURE;
    }

    const char* file = argv[1];
    float tol = 1e-6f;
    if (argc == 3) {
        tol = std::atof(argv[2]);
    }

    std::cout << "[*] Building graph..." << std::endl;
    const auto graph = Graph(file);
    std::cout << "        Nodes: " << graph.num_nodes << std::endl;

    const auto results = pagerank(graph, tol);
    const auto iterations = results.first;
    const auto ranks = results.second;
    std::cout << "[*] Ranks (after " << iterations << " iterations):" << std::endl;
    for (const auto pair: ranks) {
        const uint_fast32_t node = pair.first;
        const float rank = pair.second;
        std::cout << "        " << std::setfill('0') << std::setw(9) << node << ": " << rank << std::endl;
    }

    std::ofstream outfile;
    outfile.open("ranks-" + std::to_string(graph.num_nodes) + ".txt");
    outfile << ranks;
    outfile.close();

    return EXIT_SUCCESS;
}
