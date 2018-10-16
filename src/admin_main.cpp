#include "AdminInterface.hpp"
#include <iostream>
#include <string>

void help() {
    std::cout << "Usage:" << std::endl;
    std::cout << "create Create new type" << std::endl;
    std::cout << "exit Exit" << std::endl;
}

int main() {
    AdminInterface ai("Types.txt");
    help();
    while (true) {
        std::cout << "> ";
        std::string str;
        std::getline(std::cin, str);
        if (str == "create") {
            ai.create_new_type();
        } else if (str == "exit") {
            return 0;
        } else {
            std::cout << "Unknown command!" << std::endl;
        }
    }
}