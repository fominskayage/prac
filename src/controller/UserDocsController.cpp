#include "UserDocsController.hpp"
#include "User.hpp"

UserDocsController::UserDocsController(std::string types_filename, std::string users_filename, 
                                    std::string documents_filename, std::string coordinations_filename)
            : reader(std::shared_ptr<IReader>(new FileReader(types_filename, users_filename, documents_filename, coordinations_filename))), 
                writer(std::shared_ptr<IWriter>(new FileWriter(types_filename, documents_filename, coordinations_filename))) {}


std::vector<std::shared_ptr<Document>> UserDocsController::get_documents(const std::shared_ptr<User> &user) const {
    std::vector<std::shared_ptr<Document>> res;
    auto docs = reader->documents();
    for (auto doc : docs) {
        if (doc->get_author()->get_login() == user->get_login()) {
            res.push_back(doc);
        }
    }

    return res;
}