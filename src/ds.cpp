#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "ds.hpp"

#include "prettyprint.hpp"


Graph::Graph(const std::string filename) {
    std::ifstream infile(filename, std::ios::in);

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        uint_fast32_t from_node, to_node;
        if (!(iss >> from_node >> to_node)) continue;

        nodes.insert(from_node);
        nodes.insert(to_node);
        edges[from_node].insert(to_node);
    }

    infile.close();

    n = 0;
    if (nodes.size() > 0) {
        const auto max_element = std::max_element(std::cbegin(nodes), std::cend(nodes));
        n = *max_element + 1;
    }

    for (size_t i = 0; i < n; ++i) {
        edges.try_emplace(i);
    }
}


CSR::CSR(const arma::mat& matrix) {
    ia.push_back(0);

    size_t nonzero = 0;
    for (size_t i = 0; i < matrix.n_rows; ++i) {
        for (size_t j = 0; j < matrix.n_cols; ++j) {
            if (matrix(i, j) != 0.0f) {
                a.push_back(matrix(i, j));
                ja.push_back(j);
                ++nonzero;
            }
        }
        ia.push_back(nonzero);
    }

    assert(a.size() == ja.size());
    assert(ia.size() == matrix.n_rows+1);
    n_rows = matrix.n_rows;
    n_cols = matrix.n_cols;
}

CSR::CSR(const Graph& graph) {
    n_rows = graph.nodes.size();
    n_cols = graph.nodes.size();

    ia.push_back(0);

    size_t edge_count = 0;
    for (const auto from_node: graph.nodes) {
        const auto& neighbors = graph.edges.at(from_node);

        for (const auto to_node: neighbors) {
            const float weight = 1.0f;
            a.push_back(weight);
            ja.push_back(to_node);
        }

        edge_count += neighbors.size();
        ia.push_back(edge_count);
    }

    assert(a.size() == ja.size());
    assert(ia.size() == graph.nodes.size()+1);
}

CSR::CSR(const std::string& filename) {
    std::ifstream infile(filename, std::ios::in | std::ios::binary);

    size_t a_size, ia_size, ja_size;
    std::sscanf(filename.c_str(), "CSR-%zd-%zd-%zd-%zd.bin", &n_rows, &a_size, &ia_size, &ja_size);

    a.resize(a_size);
    infile.read(reinterpret_cast<char*>( &a[0]),  a_size * sizeof(float));
    ia.resize(ia_size);
    infile.read(reinterpret_cast<char*>(&ia[0]), ia_size * sizeof(uint_fast32_t));
    ja.resize(ja_size);
    infile.read(reinterpret_cast<char*>(&ja[0]), ja_size * sizeof(uint_fast32_t));

    infile.close();

    n_rows = 0;
    n_cols = 0;
}


arma::vec CSR::operator*(const arma::vec& vec) {
    // see http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html
    arma::vec x = arma::zeros<arma::vec>(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        if (ia[i] == ia[i+1]) {
            // for (size_t k = 0; k < n; ++k) {
            //     x[i] += (1.0f/n) * vec[k];
            // }
        }
        else {
            for (size_t k = ia[i]; k < ia[i+1]; ++k) {
                x[i] = x[i] + a[k] * vec[ja[k]];
            }
        }
    }
    return x;
}

void CSR::to_file() {
    std::stringstream filename;
    filename << "CSR-" << n_rows << "-" << a.size() << "-" << ia.size() << "-" << ja.size() << ".bin";

    std::ofstream outfile(filename.str(), std::ios::out | std::ios::binary);

    outfile.write(reinterpret_cast<const char*>( &a[0]),  a.size() * sizeof(float));
    outfile.write(reinterpret_cast<const char*>(&ia[0]), ia.size() * sizeof(uint_fast32_t));
    outfile.write(reinterpret_cast<const char*>(&ja[0]), ja.size() * sizeof(uint_fast32_t));

    outfile.close();
}
