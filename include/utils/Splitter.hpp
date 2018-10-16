#ifndef PRAC_INCLUDE_UTILS_SPLITTER_HPP
#define PRAC_INCLUDE_UTILS_SPLITTER_HPP


#include <string>
#include <vector>

class Splitter {
public:
    Splitter(char);
    std::vector<std::string> split(const std::string &) const;
private:
    char splitter;
};

#endif