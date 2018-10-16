#include "gtest/gtest.h"
#include "CommandParser.hpp"
#include "Visitor.hpp"

TEST(CommandParserTest, EmptyCommand) {
    std::stringstream str("");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, BadComma) {
    std::stringstream str("[     ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, AttrOnly) {
    std::stringstream str("Attr1");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    EvaluateVisitor evaluate_visitor;
    AttrVisitor attr_visitor;
    StringGenVisitor str_gen_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    std::map<std::string, int> vals;
    ASSERT_EQ(res->get_name(), "Attr1");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
    ASSERT_EQ("", str_gen_visitor.string_gen(*(res->get_condition())));
}

TEST(CommandParserTest, AttrAttr) {
    std::stringstream str("[ $ Attr1 == $ Attr2 ] => Attr3");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    ASSERT_EQ(res->get_name(), "Attr3");
    ASSERT_EQ(attrs.size(), 2);
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
}

TEST(CommandParserTest, DisjAttrGENE) {
    std::stringstream str("[ ( $ Attr1 >= $ Attr2 ) | ( $ Attr1 != 1 ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    StringGenVisitor str_gen_visitor;
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    ASSERT_EQ(attrs.size(), 2);
    ASSERT_EQ(res->get_name(), "Attr4");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
    ASSERT_EQ("[ ( ( ( ( ( ( ( $ Attr1 >= $ Attr2 ) ) ) ) ) | ( ( ( ( ( $ Attr1 != 1 ) ) ) ) ) ) ) ] => ", str_gen_visitor.string_gen(*(res->get_condition())));
}

TEST(CommandParserTest, DisjAttrLENE) {
    std::stringstream str("[ ( $ Attr1 <= $ Attr2 ) | ( $ Attr1 != 1 ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    ASSERT_EQ(attrs.size(), 2);
    ASSERT_EQ(res->get_name(), "Attr4");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
    StringGenVisitor str_gen_visitor;
    ASSERT_EQ("[ ( ( ( ( ( ( ( $ Attr1 <= $ Attr2 ) ) ) ) ) | ( ( ( ( ( $ Attr1 != 1 ) ) ) ) ) ) ) ] => ", str_gen_visitor.string_gen(*(res->get_condition())));
}

TEST(CommandParserTest, DisjAttrLGNeg) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2 ) | ! ( $ Attr1 > 1 ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    ASSERT_EQ(attrs.size(), 2);
    ASSERT_EQ(res->get_name(), "Attr4");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
    StringGenVisitor str_gen_visitor;
    ASSERT_EQ("[ ( ( ( ( ( ( ( $ Attr1 < $ Attr2 ) ) ) ) ) | ( ( ! ( ( ( ( $ Attr1 < 1 ) ) ) ) ) ) ) ) ] => ", str_gen_visitor.string_gen(*(res->get_condition())));
}

TEST(CommandParserTest, DisjAttrValLGNeg) {
    std::stringstream str("[ ( $ Attr1 < 10 ) | ! ( $ Attr2 >= 1 ) -> ( $ Attr1 <= 20 ) & ( $ Attr2 == 0 ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    ASSERT_EQ(attrs.size(), 2);
    ASSERT_EQ(res->get_name(), "Attr4");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
    StringGenVisitor str_gen_visitor;
    ASSERT_EQ("[ ( ( ( ( ( ( ( $ Attr1 > 10 ) ) ) ) ) | ( ( ! ( ( ( ( $ Attr2 <= 1 ) ) ) ) ) ) ) -> ( ( ( ( ( ( $ Attr1 >= 20 ) ) ) ) & ( ( ( ( $ Attr2 == 0 ) ) ) ) ) ) ) ] => ", str_gen_visitor.string_gen(*(res->get_condition())));
}

TEST(CommandParserTest, DisjAttrAttrNEG) {
    std::stringstream str("[ ( $ Attr1 != $ Attr2 ) | ( $ Attr1 > $ Attr3 ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    EvaluateVisitor evaluate_visitor;
    std::map<std::string, int> vals;
    vals["Attr1"] = 10;
    vals["Attr2"] = 10;
    vals["Attr3"] = 1;
    ASSERT_EQ(attrs.size(), 3);
    ASSERT_EQ(res->get_name(), "Attr4");
    ASSERT_EQ(true, evaluate_visitor.value(*(res->get_condition()), vals));
}

TEST(CommandParserTest, ConjAttr) {
    std::stringstream str("[ ( $ Attr1 > $ Attr2 ) & ( $ Attr1 == $ Attr3 ) -> ( $ Attr1 != $ Attr3 ) ] => Attr5");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    ASSERT_EQ(res->get_name(), "Attr5");
    StringGenVisitor str_gen_visitor;
    ASSERT_EQ("[ ( ( ( ( ( ( ( $ Attr1 > $ Attr2 ) ) ) ) & ( ( ( ( $ Attr1 == $ Attr3 ) ) ) ) ) ) -> ( ( ( ( ( ( $ Attr1 != $ Attr3 ) ) ) ) ) ) ) ] => ", str_gen_visitor.string_gen(*(res->get_condition())));

}

TEST(CommandParserTest, ImpAttr) {
    std::stringstream str("[ ( $ Attr1 <= $ Attr2 ) -> ( 10 < $ Attr3 ) ] => Attr5");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    ASSERT_EQ(res->get_name(), "Attr5");
}

TEST(CommandParserTest, CmpException) {
    std::stringstream str("[ ( $ Attr1 <> $ Attr2 ) -> ( 10 < $ Attr3 ) ] => Attr5");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, ValValException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2 ) -> ( 10 < 1 ) ] => Attr5");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, AttrValException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2 ) -> ( $ Att2 < abc ) ] => Attr5");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, AttrException) {
    std::stringstream str("[ ( $     ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, CommaException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2    ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, NegAttr) {
    std::stringstream str("[ ! ( $ Attr1 >= $ Attr2 ) | ! ( ! ( $ Attr1 != 1 ) ) ] => Attr4");
    CommandParser command_parser(str);
    std::shared_ptr<Attribute> res = command_parser.parse();
    ASSERT_EQ(res->get_name(), "Attr4");
    AttrVisitor attr_visitor;
    std::set<std::string> attrs = attr_visitor.attributes(*(res->get_condition()));
    ASSERT_EQ(attrs.size(), 2);
}

TEST(CommandParserTest, NoArrowException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2  ) ] ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}

TEST(CommandParserTest, NotArrowException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2  ) ] <> ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}


TEST(CommandParserTest, NotEndException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2  ) ] => Attr4 <> ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}


TEST(CommandParserTest, NotCommaException) {
    std::stringstream str("[ ( $ Attr1 < $ Attr2  ) <> => Attr4 <> ");
    CommandParser command_parser(str);
    ASSERT_THROW(std::shared_ptr<Attribute> res = command_parser.parse(), ParseException);
}