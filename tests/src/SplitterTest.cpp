#include "gtest/gtest.h"
#include "Splitter.hpp"
#include <vector>
#include <string>


TEST(SplitterTest, EmptyInput) {
    std::string input = "";
    Splitter splitter('.');
    std::vector<std::string> res = splitter.split(input);
    ASSERT_EQ(1, res.size());
    ASSERT_EQ(0, res[0].size());
}

TEST(SplitterTest, Singleton) {
    std::string input = "abc";
    Splitter splitter('.');
    std::vector<std::string> res = splitter.split(input);
    ASSERT_EQ(1, res.size());
    ASSERT_EQ("abc", res[0]);
}


TEST(SplitterTest, EmptyString) {
    std::string input = "..";
    Splitter splitter('.');
    std::vector<std::string> res = splitter.split(input);
    ASSERT_EQ(3, res.size());
    ASSERT_EQ(0, res[0].size());
    ASSERT_EQ(0, res[1].size());
    ASSERT_EQ(0, res[2].size());
}

