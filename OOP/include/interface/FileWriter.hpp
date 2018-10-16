#ifndef PRAC_INCLUDE_INTERFACE_FILEWRITER_HPP
#define PRAC_INCLUDE_INTERFACE_FILEWRITER_HPP


#include <memory>
#include <string>
#include <vector>

class Document;
class Type;

class IWriter {
public:
    virtual void document(std::shared_ptr<Document>) const = 0;
    virtual void type(std::shared_ptr<Type>) const = 0;
    virtual void coordination(std::pair<int, std::string>, bool) const = 0;
    virtual ~IWriter() = default;
};

class FileWriter: public IWriter {
public:
    FileWriter(std::string types_filename, std::string documents_filename = "", std::string coordinations_filename = "");
    void document(std::shared_ptr<Document>) const;
    void type(std::shared_ptr<Type>) const;
    void coordination(std::pair<int, std::string>, bool) const;
private:
    std::string types_filename;
    std::string documents_filename;
    std::string coordinations_filename;
};

#endif