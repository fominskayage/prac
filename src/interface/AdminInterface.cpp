#include "AdminInterface.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include "Type.hpp"
#include "TypeCreateController.hpp"


AdminInterface::AdminInterface(std::string filename) : type_create_controller(filename) {}

void AdminInterface::create_new_type() {
    std::cout << "Enter type name: " << std::endl;
    std::string type_name;
    std::getline(std::cin, type_name);
    type_create_controller.set_res_type_name(type_name);
    std::cout << "Enter    Attribute_Name    or    [ conditions ] => Attribute_Name    : " << std::endl;
    std::string line;
    while (std::getline(std::cin, line) && (line != "")) {
        std::stringstream str(line);
        if (!type_create_controller.check(str)) {
            std::cout << "Wrong command, try again: " << std::endl;

        }
        else {
            std::cout << "Enter    Attribute_Name    or    [ conditions ] => Attribute_Name : " << std::endl;
        }
    }
    std::cout << "Type created!" << std::endl;
    type_create_controller.save();

}