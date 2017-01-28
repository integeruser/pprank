#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <utility>

#include "utils.hpp"

#include "armadillo"
#include "prettyprint.hpp"


std::pair<uint_fast32_t, std::map<uint_fast32_t, float>> pagerank(const Graph& graph, const float tol=1E-6f)
{
    // initialization
    const uint_fast32_t n = graph.num_nodes;

    // build the adjacency matrix
    std::cout << "[*] Building adjacency matrix..." << std::endl;
    const CSC A = CSC(graph);
    const CSR At = transpose(A);

    arma::fvec p, p_new(n);
    p_new.fill(1.0f/n);

    // find dangling nodes
    std::cout << "[*] Finding dangling nodes..." << std::endl;
    arma::fvec dangling(n);
    const arma::uvec dangling_nodes =
        arma::conv_to<arma::uvec>::from(
            std::vector<uint_fast32_t>(graph.dangling_nodes.cbegin(), graph.dangling_nodes.cend()));

    const arma::fvec ones(n, arma::fill::ones);
    const float d = 0.85f;

    // ranks computation
    uint_fast32_t iterations = 0;
    std::cout << "[*] Starting PageRank..." << std::endl;
    do {
        ++iterations;
        std::cout << "        Iteration #" << iterations << "..." << std::endl;

        p = p_new;

        dangling.fill(1.0f/n * arma::sum(p(dangling_nodes)));

        p_new = (1-d)/n * ones + d * (At*p + dangling);
    }
    while (arma::norm(p_new-p, 1) >= tol);

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

    std::ofstream outfile;
    outfile.open("ranks-" + std::to_string(graph.num_nodes) + ".txt");
    outfile << ranks;
    outfile.close();

    return EXIT_SUCCESS;
}
