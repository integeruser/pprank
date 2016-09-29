#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"


arma::vec mul(const Graph& graph, const arma::vec& b, size_t n) {
    assert(graph.n == n);
    assert(b.size() == n);

    // build dense matrix of graph
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

    return A*b;
}


arma::vec mul(const CSR& A, const arma::vec& b, size_t n) {
    assert(b.size() == n);

    // see http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html

    arma::vec x = arma::zeros<arma::vec>(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i) {
        if (A.ia[i] == A.ia[i+1]) {
            // for (size_t k = 0; k < n; ++k) {
            //     x[i] += (1.0f/n) * b[k];
            // }
        }
        else {
            for (size_t k = A.ia[i]; k < A.ia[i+1]; ++k) {
                x[i] = x[i] + A.a[k] * b[A.ja[k]];
            }
        }
    }

    return x;
}


int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    // load the graph
    const auto filename = argv[1];
    const auto graph = Graph(filename);
    std::cout << "Nodes: " << graph.nodes << std::endl;
    std::cout << "Edges: " << graph.edges << std::endl;

    // build the CSR representation
    const auto csr = CSR(graph);
    std::cout << "CSR.a:  " << csr.a << std::endl;
    std::cout << "CSR.ia: " << csr.ia << std::endl;
    std::cout << "CSR.ja: " << csr.ja << std::endl;


    // multiplicate both representations for vector b and compare results 
    const auto n = graph.n;

    const auto b = arma::vec(n).randn();
    std::cout << "b: " << b << std::endl;

    const auto res1 = mul(graph, b, n);
    std::cout << "res1: " << res1 << std::endl;

    const auto res2 = mul(csr, b, n);
    std::cout << "res2: " << res2 << std::endl;

    return EXIT_SUCCESS;
}
