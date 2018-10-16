#ifndef PRAC_INCLUDE_MODEL_TYPE_HPP
#define PRAC_INCLUDE_MODEL_TYPE_HPP

#include <memory>
#include <vector>
#include <string>
class Attribute;

class Type
{
public:
    Type(std::string, std::vector<std::shared_ptr<Attribute>>);
    std::string get_type_name() const;
    std::vector<std::shared_ptr<Attribute>> get_attributes() const;
private:
    std::string type_name;
    std::vector<std::shared_ptr<Attribute>> attributes;
    
};

#endif