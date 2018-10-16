#include "ViewTypesController.hpp"


ViewTypesController::ViewTypesController(std::string types_filename, std::string users_filename,
                     std::string documents_filename, std::string coordinations_filename) 
            : reader(std::shared_ptr<FileReader>(new FileReader(types_filename, 
                                                            users_filename, 
                                                            documents_filename, 
                                                            coordinations_filename))) {}


std::vector<std::string> ViewTypesController::read_types() const {
    return reader->strings();
}