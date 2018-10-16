#ifndef PRAC_INCLUDE_MODEL_ATTRIBUTE_HPP
#define PRAC_INCLUDE_MODEL_ATTRIBUTE_HPP

#include <string>
class Formula;

class Attribute
{
public:
    Attribute(std::string name, std::shared_ptr<Formula>);
    std::string get_name() const;
    std::shared_ptr<Formula> get_condition() const;
private:
    std::string name;
    std::shared_ptr<Formula> condition;
};

#endif