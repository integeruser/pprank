#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include "ds.hpp"

#include "prettyprint.hpp"


float dist(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    const auto n = a.size();

    float d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        d += std::pow(a[i]-b[i], 2);
    }
    return std::sqrt(d);
}

std::map<uint_fast32_t, float> rank(const Graph& graph) {
    // initialization
    const auto n = graph.edges.size();

    const auto d = 0.85f;

    std::vector<float> p(n), p_new(n, 1.0f/n);

    // PageRank computation
    // to avoid storing A in memory (could not fit), recompute its rows at every iteration
    size_t iterations = 0;
    do {
        iterations += 1;

        p = p_new;

        std::fill(p_new.begin(), p_new.end(), 0.0f);
        for (size_t i = 0; i < n; ++i) {
            const auto outdegree = graph.edges.at(i).size();
            if (outdegree == 0) {
                // dangling node
                for (size_t j = 0; j < n; ++j) {
                    p_new[j] += (1.0f/n) * p[i];
                }
            }
            else {
                for (const auto j: graph.edges.at(i)) {
                    p_new[j] += (1.0f/outdegree) * p[i];
                }
            }
        }

        for (size_t i = 0; i < n; ++i) {
            p_new[i] = (1.0f-d)/n + d * p_new[i];
        }
    } while (dist(p, p_new) >= 1E-6f);
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
