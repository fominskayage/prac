#ifndef PRAC_INCLUDE_CONTROLLER_DOCCREATECONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_DOCCREATECONTROLLER_HPP


#include <string>
#include <memory>
#include "FileReader.hpp"
#include "FileWriter.hpp"
#include "Type.hpp"
#include "User.hpp"
#include "Document.hpp"

class DocCreateController
{
public:
    DocCreateController(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    std::shared_ptr<Type> get_type(std::string type_name) const;
    std::shared_ptr<User> get_user(std::string login) const;
    void save(const Document &) const;
private:
    std::shared_ptr<IReader> reader;
    std::shared_ptr<IWriter> writer;
};

#endif