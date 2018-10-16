#include "gtest/gtest.h"
#include "FileReader.hpp"
#include "Document.hpp"
#include "User.hpp"
#include "Type.hpp"

TEST(FileReaderTest, EmptyFile) {
    
}

TEST(FileReaderTest, FileNotFound) {

}

TEST(FileReaderTest, EmptyDocument) {

}

TEST(FileReaderTest, TypeReader) {
    FileReader file_reader("tests/documents/types.txt", "", "", "");
    std::vector<std::shared_ptr<Type>> types = file_reader.types();
    ASSERT_EQ(1, types.size());
    auto attr = types[0]->get_attributes();
    std::string name = types[0]->get_type_name();
    ASSERT_EQ(3, attr.size());
    ASSERT_EQ("type1", name);
}

TEST(FileReaderTest, CoordinatorsOnly) {
    FileReader file_reader("", "", "tests/documents/oneattronecoordinator.txt",  "");
    std::vector<std::shared_ptr<Document>> docs = file_reader.documents();
    auto attr = docs[0]->get_attributes();
    auto coord = docs[0]->get_coordinators();
    std::string login = docs[0]->get_author()->get_login();
    int id = docs[0]->get_id();
    ASSERT_EQ(1, docs.size());
    ASSERT_EQ("Galya", login);
    ASSERT_EQ(38, id);
    ASSERT_EQ(1, attr.size());
    ASSERT_EQ(1, coord.size());
}
