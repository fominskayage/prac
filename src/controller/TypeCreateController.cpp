
#include "TypeCreateController.hpp"
#include "CommandParser.hpp"
#include "Visitor.hpp"

TypeCreateController::TypeCreateController(std::string str) 
        : writer(std::shared_ptr<IWriter>(new FileWriter(str))) {}

bool TypeCreateController::check(std::stringstream &str) {
    try {
        CommandParser parser = CommandParser(str);
        std::shared_ptr<Attribute> attr = parser.parse();
        AttrVisitor attr_visitor;
        std::set<std::string> cur_attr_names = attr_visitor.attributes(*(attr->get_condition()));
        for (std::string str : cur_attr_names) {
            if (attr_names.count(str) == 0) {
                throw ParseException();///на самом деле не parse а semantic
            }
        }
        res_attributes.push_back(attr);
        attr_names.insert(attr->get_name());
        return true;

    } catch (ParseException &) {
        return false;
    }
}

void TypeCreateController::set_res_type_name(std::string str) {
    res_type_name = str;
}

void TypeCreateController::save() {
    Type *res_type = new Type(res_type_name, res_attributes);
    writer->type((std::shared_ptr<Type>)res_type);
}