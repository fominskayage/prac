#ifndef PRAC_INCLUDE_CONTROLLER_COORDINATECONTROLLER_HPP
#define PRAC_INCLUDE_CONTROLLER_COORDINATECONTROLLER_HPP

#include <memory>
#include <vector>
#include <map>

class User;
class Document;
class IReader;
class IWriter;


class CoordinateController
{
public:
    CoordinateController(std::shared_ptr<User> &user, std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    std::vector<std::shared_ptr<Document>> get_documents() const;
    std::map<std::pair<int, std::string>, bool> get_coordinations() const;
    void coordinate(int doc_id, bool) const;
private:
    std::shared_ptr<User> &user;
    std::shared_ptr<IReader> reader;
    std::shared_ptr<IWriter> writer;
};

#endif

