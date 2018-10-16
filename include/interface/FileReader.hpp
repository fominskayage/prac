#ifndef PRAC_INCLUDE_INTERFACE_FILEREADER_HPP
#define PRAC_INCLUDE_INTERFACE_FILEREADER_HPP


#include <memory>
#include <string>
#include <vector>
#include <map>

class Document;
class Type;
class User;


class IReader {
public:
    virtual std::vector<std::shared_ptr<Document> > documents() const = 0;
    virtual std::vector<std::shared_ptr<User> > users() const = 0;
    virtual std::vector<std::shared_ptr<Type> > types() const = 0;
    virtual std::map<std::pair<int, std::string>, bool> coordinations() const = 0;
    virtual std::vector<std::string > strings() const = 0;
    virtual ~IReader() = default;
};

class FileReader: public IReader {
public:
    FileReader(std::string types_filename, std::string users_filename, std::string documents_filename, std::string coordinations_filename);
    std::vector<std::shared_ptr<Document> > documents() const;
    std::vector<std::shared_ptr<User> > users() const;
    std::vector<std::shared_ptr<Type> > types() const;
    std::map<std::pair<int, std::string>, bool> coordinations() const;
    std::vector<std::string > strings() const;
private:
    std::string types_filename;
    std::string users_filename;
    std::string documents_filename;
    std::string coordinations_filename;

};



#endif