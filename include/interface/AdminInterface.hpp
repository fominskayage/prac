#ifndef PRAC_INCLUDE_INTERFACE_ADMININTERFACE_HPP
#define PRAC_INCLUDE_INTERFACE_ADMININTERFACE_HPP

#include <memory>
#include "TypeCreateController.hpp"

class AdminInterface
{
public:
    AdminInterface(std::string filename);
    void create_new_type();
private:
    TypeCreateController type_create_controller;
};

#endif