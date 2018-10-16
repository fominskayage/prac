#include "UserInterface.hpp"
#include <iostream>
#include <string>

void help() {
    std::cout << "Usage:" << std::endl;
    std::cout << "login Login" << std::endl;
    std::cout << "logout Logout" << std::endl;
    std::cout << "create Create new document" << std::endl;
    std::cout << "coordinate Coordinate documents" << std::endl;
    std::cout << "view View my documents" << std::endl;
    std::cout << "view_types View types" << std::endl;
    std::cout << "view_to_coord View documents to coordinate" << std::endl;
    std::cout << "exit Exit" << std::endl;
}

int main() {
    UserInterface ui("types.txt", "users.txt", "documents.txt", "coordinations.txt");
    help();
    while (true) {
        std::cout << "> ";
        std::string str;
        std::getline(std::cin, str);
        if (str == "login") {
            ui.login();
        } else if (str == "logout") {
            ui.logout();
        } else if (str == "view") {
            ui.view_my_documents();
        } else if (str == "create") {
            ui.create_new_document();
        } else if (str == "coordinate") {
            ui.coordinate_document();
        } else if (str == "view_types") {
            ui.view_types();
        } else if (str == "exit") {
            return 0;
        } else {
            std::cout << "Unknown command!" << std::endl;
        }
    }
}