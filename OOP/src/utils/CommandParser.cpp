#include <iostream>
#include "CommandParser.hpp"
#include "Formula.hpp"
/*  
    S -> Cond => Attr | Attr
    Cond ->[Imp]
    Imp -> Disj { '->' Disj }
    Disj -> Conj { '|' Conj }
    Conj -> Neg { '&' Neg }
    Neg -> ! Neg | Expr
    Expr -> ( Imp ) | Attr Cmp Val | Attr Cmp Attr | Val Cmp Attr
    Attr -> $ {1..9, a..z , A..Z, _}
    Val -> 1 Num | 2 Num ... | 9 Num
    Num -> 0 Num | ... | 9 Num
    Cmp -> >= | <= | == | < | > | !=
*/

CommandParser::CommandParser(std::stringstream &source) 
        : source(std::move(source)) {}


std::shared_ptr<Attribute> CommandParser::parse() {
    get_token();
    std::shared_ptr<Formula> res_cond;
    std::string res_attr_name;
    if (token == "[") {
        get_token();
        res_cond = ImpF();
        if (token != "]") {
            throw ParseException();
        }
        get_token();
        if (token != "=>") {
            throw ParseException();
        }
        get_token();
        res_attr_name = token;
        get_token();
    }
    else if (token != "") {
        TrueFormula *v = new TrueFormula();
        Formula *tmp = v;
        res_cond = std::shared_ptr<Formula> (tmp);
        res_attr_name = token;
        get_token();
    }
    else {
        throw ParseException();
    }
    if (token != "") {
        throw ParseException();
    }
    Attribute *v = new Attribute(res_attr_name, res_cond);
    std::shared_ptr<Attribute> res = std::shared_ptr<Attribute> (v);
    return res;
}

std::shared_ptr<Formula> CommandParser::ImpF() {
    ImplicationFormula *v = new ImplicationFormula();
    Formula *tmp = v;
    std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
    v->add(DisjF());
    while (token == "->") {
        get_token();
        v->add(DisjF());
    }
    return std::shared_ptr<Formula> (res);
}

std::shared_ptr<Formula> CommandParser::DisjF() {
    DisjunctionFormula *v = new DisjunctionFormula();
    Formula *tmp = v;
    std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
    v->add(ConjF());
    while (token == "|") {
        get_token();
        v->add(ConjF());
    }
    return std::shared_ptr<Formula> (res);
}

std::shared_ptr<Formula> CommandParser::ConjF() {
    ConjunctionFormula *v = new ConjunctionFormula();
    Formula *tmp = v;
    std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
    v->add(NegF());
    while (token == "&") {
        get_token();
        v->add(NegF());
    }
    return res;
}

std::shared_ptr<Formula> CommandParser::NegF() {
    if (token == "!") {
        get_token();
        NegationFormula *v = new NegationFormula(NegF());
        Formula *tmp = v;
        std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
        return res;
    }
    else {
        std::shared_ptr<Formula> res = ExprF();
        return res;
    }
}

std::shared_ptr<Formula> CommandParser::ExprF() {
    if (token == "(") {
        get_token();
        std::shared_ptr<Formula> res = ImpF();
        if (token != ")") {
            throw ParseException();
        }
        get_token();
        return res;
    } 
    else
    {
        if (token == "$") {
            get_token();
            if (token == "") {
                throw ParseException();
            }
            std::string attr1 = token;
            get_token();
            Formula::Cmp cmp = Cmp();
            if (token == "$") {
                get_token();
                std::string attr2 = token;
                get_token();
                AttrAttrExpressionFormula *v = new AttrAttrExpressionFormula(attr1, attr2, cmp);
                Formula *tmp = v;
                std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
                return res;
            }
            else {
                int val;
                if (cmp == Formula::G) {
                    cmp = Formula::L;
                }
                else if (cmp == Formula::L) {
                    cmp = Formula::G;
                }
                else if (cmp == Formula::LE) {
                    cmp = Formula::GE;
                }
                else if (cmp == Formula::GE) {
                    cmp = Formula::LE;
                }
                try {
                    val = std::stoi(token);
                } catch (std::invalid_argument) {
                    throw ParseException();
                }
                get_token();
                ConstAttrExpressionFormula *v = new ConstAttrExpressionFormula(attr1, val, cmp);
                Formula *tmp = v;
                std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
                return res;
            }
        }
        else {
            int val;
            try {
                val = std::stoi(token);
            } catch (std::invalid_argument) {
                throw ParseException();
            }
            get_token();
            Formula::Cmp cmp = Cmp();
            if (token != "$") {
                throw ParseException();
            }
            get_token();
            std::string attr2 = token;
            get_token();
            ConstAttrExpressionFormula *v = new ConstAttrExpressionFormula(attr2, val, cmp);
            Formula *tmp = v;
            std::shared_ptr<Formula> res = std::shared_ptr<Formula> (tmp);
            return res;
        }
    }
}


Formula::Cmp CommandParser::Cmp() {
    if (token == ">=") {
        get_token();
        return Formula::GE;
    }
    else if (token == ">") {
        get_token();
        return Formula::G;
    }
    else if (token == "<=") {
        get_token();
        return Formula::LE;
    }
    else if (token == "<") {
        get_token();
        return Formula::L;
    }
    else if (token == "==") {
        get_token();
        return Formula::E;
    }
    else if (token == "!=") {
        get_token();
        return Formula::NE;
    }
    else
    {
        throw ParseException();
    }
}


void CommandParser::get_token() {
    std::string res = "";
    int c;
    c = source.get();
    if (c == EOF) {
        token = res;
        return;
    }
    while (isspace(c)) {
        c = source.get();
    }
    while (!isspace(c) && (c != EOF)) {
        res.push_back(c);
        c = source.get();
    }
    token = res;

}
