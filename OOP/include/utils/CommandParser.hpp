#ifndef PRAC_INCLUDE_UTILS_COMMANDPARSER_HPP
#define PRAC_INCLUDE_UTILS_COMMANDPARSER_HPP

/*    S -> Cond => Attr | Attr
    Cond ->[Imp]
    Imp -> Disj { '->' Disj }
    Disj -> Conj { '|' Conj }
    Conj -> Neg { '&' Neg }
    Neg -> ! Neg | Expr
    Expr -> ( Disj ) | Attr Cmp Val | Attr Cmp Attr | Val Cmp Attr
    Attr -> $ {1..9, a..z , A..Z, _}
    Val -> 1 Num | 2 Num ... | 9 Num
    Num -> 0 Num | ... | 9 Num
    Cmp -> >= | <= | == | < | > | !=

*/
#include <string>
#include <sstream>
#include <memory>
#include "Formula.hpp"
#include "Attribute.hpp"

class ParseException {
};

class CommandParser {
public:
    CommandParser(std::stringstream &);
    std::shared_ptr<Attribute> parse();
private:
    std::shared_ptr<Formula> ImpF();
    std::shared_ptr<Formula> DisjF();
    std::shared_ptr<Formula> ConjF();
    std::shared_ptr<Formula> NegF();
    std::shared_ptr<Formula> ExprF();
    Formula::Cmp Cmp();
    void get_token();
    std::string token;
    std::stringstream source;
};

#endif