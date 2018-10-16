#include "LoginController.hpp"
#include "User.hpp"
#include "FileReader.hpp"

LoginController::LoginController(std::string types_filename, 
                                std::string users_filename, 
                                std::string documents_filename, 
                                std::string coordinations_filename) 
        : reader(std::shared_ptr<FileReader>(new FileReader(types_filename, 
                                                            users_filename, 
                                                            documents_filename, 
                                                            coordinations_filename))) {}

std::shared_ptr<User> LoginController::get_user(std::string login) {
    std::vector<std::shared_ptr<User>> users = reader->users();
    for (std::shared_ptr<User> user : users) {
        if (user->get_login() == login) {
            return user;
        }
    }
    return nullptr;
}