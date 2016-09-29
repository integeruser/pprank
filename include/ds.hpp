#ifndef DS_HPP
#define DS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>


struct Graph {
    size_t n;
    std::set<uint_fast32_t> nodes;
    std::map<uint_fast32_t, std::set<uint_fast32_t>> edges;

    Graph(const std::string filename);
};


struct CSR {
    size_t n;
    size_t n_rows, n_cols;
    std::vector<float> a;
    std::vector<uint_fast32_t> ia, ja;

    CSR();
    CSR(const Graph& graph);
    CSR(const std::string filename);

    void to_file();
};


#endif
