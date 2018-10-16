#ifndef PRAC_INCLUDE_CONTROLLER_VIEWTYPESCONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_VIEWTYPESCONTROLLER_HPP

#include <memory>
#include <string>
#include "FileReader.hpp"

class ViewTypesController
{
public:
    ViewTypesController(std::string types_filename, std::string users_filename = "", std::string documents_filename = "", std::string coordinations_filename = "");
    std::vector<std::string> read_types() const;
private:
    std::shared_ptr<IReader> reader;
};
#endif