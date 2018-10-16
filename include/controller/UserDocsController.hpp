#ifndef PRAC_INCLUDE_CONTROLLER_USERDOCSCONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_USERDOCSCONTROLLER_HPP

#include "FileReader.hpp"
#include "FileWriter.hpp"
#include "Document.hpp"

class UserDocsController
{
public:
    UserDocsController(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    std::vector<std::shared_ptr<Document>> get_documents(const std::shared_ptr<User> &user) const;
private:
    std::shared_ptr<IReader> reader;
    std::shared_ptr<IWriter> writer;

};

#endif