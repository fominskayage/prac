#include "Splitter.hpp"

#include <iostream>

Splitter::Splitter(char splitter) : splitter(splitter) {}

std::vector<std::string> Splitter::split(const std::string & str) const {
    std::vector<std::string> res;

    size_t pos = 0;
    size_t find;
    while ( (find = str.find(splitter, pos)) != std::string::npos) {
        res.push_back(str.substr(pos, find - pos));
        pos = find + 1;
    }
    res.push_back(str.substr(pos));

    return res;
}