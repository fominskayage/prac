#ifndef PRAC_INCLUDE_INTERFACE_USERINTERFACE_HPP
#define PRAC_INCLUDE_INTERFACE_USERINTERFACE_HPP

#include "UserDocsController.hpp"
#include "DocCreateController.hpp"
#include "ViewTypesController.hpp"
#include "CoordinateController.hpp"
#include "LoginController.hpp"
#include <memory>

class UserInterface
{
public:
    UserInterface(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    void login();
    void logout();
    void create_new_document() const;
    void coordinate_document() const;
    void view_my_documents() const;
    void view_types() const;

private:
    std::shared_ptr<User> user;
    LoginController login_controller;
    CoordinateController coordinate_controller;
    ViewTypesController view_types_controller;
    DocCreateController doc_create_controller;
    UserDocsController user_docs_controller;
};

#endif