#include "Document.hpp"
#include <iostream>

Document::Document(std::shared_ptr<User> author, std::map<std::string, int> &attributes, std::vector<std::shared_ptr<User>> &coordinators, int id) 
            : author(author), attributes(attributes), coordinators(coordinators), id(id) {}


std::shared_ptr<User> Document::get_author() const {
    return author;
}

std::map<std::string, int> Document::get_attributes() const {
    return attributes;
}

std::vector<std::shared_ptr<User>> &Document::get_coordinators() {
    return coordinators;
}

int Document::get_id() const {
    return id;
}