#ifndef PRAC_INCLUDE_CONTROLLER_TYPECREATECONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_TYPECREATECONTROLLER_HPP

#include <string>
#include <memory>
#include <sstream>
#include <vector>
#include <set>
#include "Attribute.hpp"
#include "FileWriter.hpp"
#include "Type.hpp"

class TypeCreateController
{
public:
    TypeCreateController(std::string types_filename);
    void set_res_type_name(std::string);
    bool check(std::stringstream &str);
    void save();
    
private:
    std::shared_ptr<IWriter> writer;
    std::string res_type_name;
    std::vector<std::shared_ptr<Attribute>> res_attributes;
    std::set<std::string> attr_names;
    
};
#endif