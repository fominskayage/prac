#include "Attribute.hpp"

Attribute::Attribute(std::string name, std::shared_ptr<Formula> cond) : name(name), condition(cond) {}

std::string Attribute::get_name() const {
    return name;
}

std::shared_ptr<Formula> Attribute::get_condition() const {
    return condition;
}