#include <cassert>
#include <iostream>
#include <vector>

#include "ds.hpp"

#include "armadillo"
#include "prettyprint.hpp"


// split a CSR matrix into n submatrices
std::vector<CSR> split(const CSR& csr, size_t n) {
    const size_t rows = csr.ia.size()-1;
    assert(0 < n and n <= rows);

    // compute maximum size of each submatrix
    // note that the last submatrix can have a smaller size than the others
    const size_t size = ceil(float(rows)/n);

    std::vector<CSR> csrs;
    size_t i = 1, j = 0;
    size_t offset = 0;
    do {
        auto subcsr = CSR();

        subcsr.ia.push_back(0);
        for (size_t k = 0; i < csr.ia.size() && k < size; ++k) {
            subcsr.ia.push_back(csr.ia[i] - offset);
            ++i;
        }
        offset += subcsr.ia.back();
        assert(((csrs.size() < n-1) and subcsr.ia.size() == size+1) or
             ((csrs.size() == n-1) and subcsr.ia.size() <= size+1));

        for (size_t k = 0; k < subcsr.ia.back(); ++k) {
            subcsr.a.push_back(csr.a[j]);
            subcsr.ja.push_back(csr.ja[j]);
            ++j;
        }

        csrs.push_back(subcsr);
    } while (csrs.size() < n);

    assert(i == csr.ia.size());
    assert(j == csr.a.size() and j == csr.ja.size());
    return csrs;
}


int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return EXIT_FAILURE;
    }

    const auto filename = argv[1];
    const auto& graph = Graph(filename);

    const auto& csr = CSR(graph);
    std::cout << "CSR.a:  " << csr.a << std::endl;
    std::cout << "CSR.ia: " << csr.ia << std::endl;
    std::cout << "CSR.ja: " << csr.ja << std::endl;

    const auto& csrs = split(csr, 3);
    for (const auto& subcsr: csrs) {
        std::cout << "subcsr.a:  " << subcsr.a << std::endl;
        std::cout << "subcsr.ia: " << subcsr.ia << std::endl;
        std::cout << "subcsr.ja: " << subcsr.ja << std::endl;
    }

    return EXIT_SUCCESS;
}
