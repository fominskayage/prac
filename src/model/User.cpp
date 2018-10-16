#include "User.hpp"

User::User(std::string login) : login(login) {}


std::string User::get_login() const {
    return login;
}
