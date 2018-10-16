#include "FileWriter.hpp"
#include <iostream>
#include <fstream>
#include <utility>
#include "Type.hpp"
#include "Visitor.hpp"
#include "Attribute.hpp"
#include "Document.hpp"
#include "User.hpp"

FileWriter::FileWriter(std::string types_filename, std::string documents_filename, 
                                                    std::string coordinations_filename) 
                : types_filename(types_filename), documents_filename(documents_filename), 
                                                coordinations_filename(coordinations_filename) {}


void FileWriter::document(std::shared_ptr<Document> doc) const {
    if (doc != nullptr) {
        std::ofstream out(documents_filename, std::ios::app);
        out << doc->get_author()->get_login() << std::endl;
        for (auto attr : doc->get_attributes()) {
            out << std::get<0>(attr) << ' ' << std::to_string(std::get<1>(attr)) << std::endl;
        }
        out << std::endl;
        for (auto coord : doc->get_coordinators()) {
            out << coord->get_login() << std::endl;
        }
        out << std::endl;
        int id = out.tellp();
        out << std::to_string(id) << std::endl;
        out << std::endl;
    }
}

void FileWriter::type(std::shared_ptr<Type> type) const {
    if (type != nullptr) {
        std::ofstream out(types_filename, std::ios::app);
        out << type->get_type_name() << std::endl;
        StringGenVisitor string_gen_visitor;
        for (auto attr : type->get_attributes()) {
            out << string_gen_visitor.string_gen(*(attr->get_condition())) << attr->get_name() << std::endl;
        }
        out << std::endl;
    }
}

void FileWriter::coordination(std::pair<int, std::string> doc_user, bool coord) const {
    std::ofstream out(coordinations_filename, std::ios::app);
    out << std::get<0>(doc_user) << ' ' << std::get<1>(doc_user) << ' ' << coord << std::endl;
}
