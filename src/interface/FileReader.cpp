#include "FileReader.hpp"
#include "Attribute.hpp"
#include "User.hpp"
#include "Document.hpp"
#include "Type.hpp"
#include "Splitter.hpp"
#include "CommandParser.hpp"

#include <iostream>
#include <vector>
#include <fstream>


FileReader::FileReader(std::string types_filename, std::string users_filename,
                     std::string documents_filename, std::string coordinations_filename)
            : types_filename(types_filename), users_filename(users_filename),
            documents_filename(documents_filename), coordinations_filename(coordinations_filename) {}

std::vector<std::shared_ptr<Document>> FileReader::documents() const {
    std::vector<std::shared_ptr<Document>> res;

    std::ifstream infile(documents_filename);
    std::string line;

    Splitter splitter(' ');

    while (std::getline(infile, line)) {
        std::shared_ptr<User> author = std::shared_ptr<User> (new User(line));
        std::map<std::string, int> attributes;
        std::vector<std::shared_ptr<User>> coordinators;
        while (std::getline(infile, line) && (line != "")) {
            std::vector<std::string> splitted = splitter.split(line);
            try {
                if (splitted.size() < 2) {
                    throw std::invalid_argument("");
                }
                std::string attr_name = splitted[0];
                int attr_value = std::stoi(splitted[1]);
                attributes[attr_name] = attr_value;
            } catch (std::invalid_argument) {
                std::cerr << "WARNING: Attribute reading error!" << std::endl;
                continue ;
            }
        }
        while (std::getline(infile, line) && (line != "")) {
            std::shared_ptr<User> coordinator = std::shared_ptr<User> (new User(line));
            coordinators.push_back(coordinator);
        }
        std::getline(infile, line);
        int doc_id = std::stoi(line);
        res.push_back(std::shared_ptr<Document> (new Document(author, attributes, coordinators, doc_id)));
        std::getline(infile, line);
    }

    return res;
}

std::vector<std::shared_ptr<User>> FileReader::users() const {
    std::vector<std::shared_ptr<User>> res;

    std::ifstream infile(users_filename);


    Splitter splitter(',');
    std::string line;

    while (std::getline(infile, line)) {
        std::vector<std::string> splitted = splitter.split(line);
        try {
            if (splitted.size() < 1) {
                throw std::invalid_argument("");
            }
            std::string login = splitted[0];

            res.push_back(std::shared_ptr<User>(new User(login)));
        } catch (std::invalid_argument) {
            std::cerr << "WARNING: User reading error!" << std::endl;
            continue ;
        }
    }

    return res;
}

std::vector<std::shared_ptr<Type>> FileReader::types() const {
    std::vector<std::shared_ptr<Type>> res;
    std::ifstream infile(types_filename);
    std::string line;
    std::getline(infile, line);
    while (std::getline(infile, line)) {
        std::string type_name = line;
        std::vector<std::shared_ptr<Attribute>> attrs;
        while (std::getline(infile, line) && (line != "")) {
            std::stringstream str(line);
            CommandParser parser(str);
            std::shared_ptr<Attribute> attr = parser.parse();
            attrs.push_back(attr);
        }
        res.push_back(std::shared_ptr<Type>(new Type(type_name, attrs)));
    }
    return res;
}


std::map<std::pair<int, std::string>, bool> FileReader::coordinations() const {
    std::map<std::pair<int, std::string>, bool> res;
    std::ifstream infile(coordinations_filename);
    std::string line;
    Splitter splitter(' ');
    while (std::getline(infile, line)) {
        std::vector<std::string> splitted = splitter.split(line);
        try {
            if (splitted.size() < 3) {
                throw std::invalid_argument("");
            }
            int doc_id = std::stoi(splitted[0]);
            std::string user_login = splitted[1];
            bool coord = std::stoi(splitted[2]);

            std::pair<int, std::string> key = std::pair<int, std::string>(doc_id, user_login);
            res[key] = coord;
            
        } catch (std::invalid_argument) {
            std::cerr << "WARNING: Coordination reading error!" << std::endl;
            continue ;
        }
    }

    return res;
}


std::vector<std::string > FileReader::strings() const {
    std::ifstream infile(types_filename);
    std::vector<std::string > res;
    std::string line;

    while (std::getline(infile, line)) {
        res.push_back(line);
    }

    return res;
}


