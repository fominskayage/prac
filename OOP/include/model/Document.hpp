#ifndef PRAC_INCLUDE_MODEL_DOCUMENT_HPP
#define PRAC_INCLUDE_MODEL_DOCUMENT_HPP

#include <memory>
#include <vector>
#include <string>
#include <map>
class User;
class Attribute;
class Type;

class Document
{
public:
    Document(std::shared_ptr<User> author, std::map<std::string, int> &attributes, std::vector<std::shared_ptr<User>> &coordinators, int id = 0);
    std::shared_ptr<User> get_author() const;
    std::map<std::string, int> get_attributes() const;
    std::vector<std::shared_ptr<User>> &get_coordinators();
    int get_id() const;

private:
    int id;
    std::shared_ptr<User> author;
    std::map<std::string, int> attributes;
    std::vector<std::shared_ptr<User>> coordinators;
    
};

#endif