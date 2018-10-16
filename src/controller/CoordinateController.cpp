#include "CoordinateController.hpp"
#include "FileReader.hpp"
#include "FileWriter.hpp"
#include <iostream>
#include <memory>
#include "Document.hpp"
#include "User.hpp"


CoordinateController::CoordinateController(std::shared_ptr<User> &user, std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename) :
            user(user), reader(std::shared_ptr<IReader>(new FileReader(types_filename, users_filename, documents_filename, coordinations_filename))),
                 writer(std::shared_ptr<IWriter>(new FileWriter(types_filename, documents_filename, coordinations_filename))) {}

std::vector<std::shared_ptr<Document>> CoordinateController::get_documents() const{
    std::vector<std::shared_ptr<Document>> res;
    auto docs = reader->documents();
    for (auto doc : docs) {
        auto coord = doc->get_coordinators();
        for (auto coordinator : coord) {
            if (coordinator == user) {
                res.push_back(doc);
            }
        }
    }
    return res;
}

std::map<std::pair<int, std::string>, bool> CoordinateController::get_coordinations() const {
    return reader->coordinations();
}


void CoordinateController::coordinate(int doc_id, bool coord) const {
    writer->coordination(std::pair<int, std::string>(doc_id, user->get_login()), coord);
}
