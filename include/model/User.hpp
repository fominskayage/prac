#ifndef PRAC_INCLUDE_MODEL_USER_HPP
#define PRAC_INCLUDE_MODEL_USER_HPP

#include <vector>
#include <memory>
#include <string>


class User {
public:
    User(std::string login);
    std::string get_login() const;


private:
    std::string login;
};

#endif
