#ifndef PRAC_INCLUDE_CONTROLLER_LOGINCONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_LOGINCONTROLLER_HPP

#include <memory>
#include <string>

class FileReader;

class User;

class LoginController {
public:
    LoginController(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    std::shared_ptr<User> get_user(std::string);
private:
    std::shared_ptr<FileReader> reader;
};

#endif