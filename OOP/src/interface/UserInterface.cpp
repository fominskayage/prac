#include "UserInterface.hpp"
#include "UserDocsController.hpp"
#include "Visitor.hpp"
#include "Attribute.hpp"
#include <iostream>
#include <string>


UserInterface::UserInterface(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename)
            : login_controller(types_filename, users_filename, documents_filename, coordinations_filename), 
                coordinate_controller(this->user, types_filename, users_filename, documents_filename, coordinations_filename), 
                view_types_controller(types_filename, users_filename, documents_filename, coordinations_filename), 
                doc_create_controller(types_filename, users_filename, documents_filename, coordinations_filename), 
                user_docs_controller(types_filename, users_filename, documents_filename, coordinations_filename) {}


void UserInterface::login() {
    std::cout << "Enter login: ";
    std::string login;
    std::getline(std::cin, login);
    std::shared_ptr<User> user = login_controller.get_user(login);
    if (user != nullptr) {
        this->user = user;
        std::cout << "User login success." << std::endl;
    } else {
        std::cout << "Login failed!" <<  std::endl;
    }
}

void UserInterface::logout() {
    user = nullptr;
    std::cout << "Logout: OK" << std::endl;
}

void UserInterface::create_new_document() const {
    if (user == nullptr) {
        std::cout << "Please, login" << std::endl;
        return;
    }
    std::cout << "Enter type name: " << std::endl;
    std::string type_name;
    std::getline(std::cin, type_name);
    std::shared_ptr<Type> type = doc_create_controller.get_type(type_name);
    EvaluateVisitor evaluate_visitor;
    std::map<std::string, int> values;
    if (type != nullptr) {
        for (auto attr : type->get_attributes()) {
            if (evaluate_visitor.value(*(attr->get_condition()), values)) {
                std::cout << "Enter value of " << attr->get_name() << std::endl;
                std::string str_value;
                std::getline(std::cin, str_value);
                int value = std::stoi(str_value);
                values[attr->get_name()] = value;
            }
        }
        std::cout << "Enter coordinators: " << std::endl;
        std::string line;
        std::vector<std::shared_ptr<User>> coordinators;
        while (getline(std::cin, line) && (line != "")) {
            std::shared_ptr<User> user = doc_create_controller.get_user(line);
            if (user != nullptr) {
                coordinators.push_back(user);
            }
            else {
                std::cout << "User not found! Try again: " << std::endl;
            }
        }
        Document res_doc = Document(this->user, values, coordinators);
        doc_create_controller.save(res_doc);
        std::cout << "Document saved" << std::endl;
    }
    else {
        std::cout << "Type not found!" << std::endl;
    }
}

void UserInterface::coordinate_document() const {
    if (user != nullptr) {
        std::vector<std::shared_ptr<Document>> docs = coordinate_controller.get_documents();
        std::map<std::pair<int, std::string>, bool> coordinations = coordinate_controller.get_coordinations();
        if (docs.size() > 0) {
            for (auto doc : docs) {
                std::cout << "Doc id = " << doc->get_id() << std::endl;
                std::cout << std::endl;
                std::cout << doc->get_author()->get_login() << std::endl;
                for (auto attr : doc->get_attributes()) {
                    std::cout << std::get<0>(attr) << " = " << std::to_string(std::get<1>(attr)) << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << "Enter id of document to coordinate: " << std::endl;
            std::cout << "> ";
            std::string input;
            int doc_id;
            try {
                std::getline(std::cin, input);
                doc_id = std::stoi(input);
                if (coordinations.count(std::pair<int, std::string>(doc_id, user->get_login())) == 0) {
                    std::cout << "Enter coordination result [1/0]: " << std::endl;
                    std::cout << "> ";
                    std::getline(std::cin, input);
                    bool coord = std::stoi(input);
                    coordinate_controller.coordinate(doc_id, coord);
                    std::cout << "Coordination saved." << std::endl;
                }
                else {
                    std::cout << "Document is already coordinated!" << std::endl;
                }
                
            } catch (std::invalid_argument) {
                std::cout << "Invalid!" << std::endl;
            }
        }
        else {
            std::cout << "You have no documents to coordinate!" << std::endl;
        }
    }
    else {
        std::cout << "Please, login!" << std::endl;
    }
}

void UserInterface::view_my_documents() const {
    if (user != nullptr) {
        std::vector<std::shared_ptr<Document>> docs = user_docs_controller.get_documents(user);
        if (docs.size() > 0) {
            for (auto doc : docs) {
                std::cout << "Doc id = " << doc->get_id() << std::endl;
                std::cout << std::endl;
                std::cout << doc->get_author()->get_login() << std::endl;
                for (auto attr : doc->get_attributes()) {
                    std::cout << std::get<0>(attr) << " = " << std::to_string(std::get<1>(attr)) << std::endl;
                }
                std::cout << std::endl;
                std::cout << "Coordinators: " << std::endl;
                for (auto coord : doc->get_coordinators()) {
                    std::cout << coord->get_login() << std::endl;
                }
                std::cout << std::endl;
            }
        }
        else {
            std::cout << "You haven't created documents!" << std::endl;
        }
    }
    else {
        std::cout << "Please, login!" <<  std::endl;
    }
}

void UserInterface::view_types() const {
    std::vector<std::string> types = view_types_controller.read_types();
    for (auto str : types) {
        std::cout << str << std::endl;
    }

}
