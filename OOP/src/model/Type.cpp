
#include "Type.hpp"

Type::Type(std::string type_name, std::vector<std::shared_ptr<Attribute>> attributes) 
            : type_name(type_name), attributes(attributes) {}


std::string Type::get_type_name() const {
    return type_name;
}


std::vector<std::shared_ptr<Attribute>> Type::get_attributes() const {
    return attributes;
}