
#include "DocCreateController.hpp"

DocCreateController::DocCreateController(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename)
        : reader(std::shared_ptr<IReader>(new FileReader(types_filename, users_filename, documents_filename, coordinations_filename))), 
            writer(std::shared_ptr<IWriter>(new FileWriter(types_filename, documents_filename, coordinations_filename))) {}

std::shared_ptr<Type> DocCreateController::get_type(std::string type_name) const {
    std::vector<std::shared_ptr<Type>> types = reader->types();
    for (auto type : types) {
        if (type->get_type_name() == type_name) {
            return type;
        }
    }
    return nullptr;
}

std::shared_ptr<User> DocCreateController::get_user(std::string login) const {
    std::vector<std::shared_ptr<User>> users = reader->users();
    for (std::shared_ptr<User> user : users) {
        if (user->get_login() == login) {
            return user;
        }
    }
    return nullptr;
}

void DocCreateController::save(const Document &doc) const {
    writer->document(std::shared_ptr<Document>(new Document(doc)));
}