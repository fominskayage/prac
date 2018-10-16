#ifndef PRAC_INCLUDE_MODEL_FORMULA_HPP
#define PRAC_INCLUDE_MODEL_FORMULA_HPP

/*    S -> Cond => Attr | Attr
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
#include <vector>
#include <memory>
#include <string>

#include "Visitable.hpp"

class Visitor;

class Visitable;


class Formula: public Visitable {
public:
    enum Cmp
    {
        G, GE, E, NE, LE, L
    };
};

class TrueFormula: public Formula
{
public:
    void accept(Visitor &) const;
};

class Operand {
public:
virtual std::string get_str() const = 0;
};

class NaryFormula: public Formula {
public:
    void add(std::shared_ptr<Formula> f) { 
        if (!f) {
            throw std::logic_error("Null pointer is added");
        }
        operands.push_back(f);
    }
private:
    std::vector<std::shared_ptr<Formula>> operands;
public:
    decltype(operands.cbegin()) begin() const { 
        return operands.cbegin(); 
    }
    decltype(operands.cend()) end() const { 
        return operands.cend(); 
    }
};




class ImplicationFormula: public NaryFormula {
public:
    void accept(Visitor &) const;

};

class ConjunctionFormula: public NaryFormula {
public:
    void accept(Visitor &) const;

};


class DisjunctionFormula: public NaryFormula {
public:
    void accept(Visitor &) const;

};

class NegationFormula: public Formula {
public:
    NegationFormula(std::shared_ptr<Formula> f) : f(f) { 
        if (!f) {
            std::logic_error("Null pointer added");
        }
    }

    const Formula &formula() const { 
        return *f; 
    }

    void accept(Visitor &) const;
// YOUR CODE
private:
    std::shared_ptr<Formula> f;
};


class AttrAttrExpressionFormula : public Formula {
public:
    AttrAttrExpressionFormula(std::string, std::string, Cmp);
    std::string get_attr1_name() const;
    std::string get_attr2_name() const;
    Cmp get_cmp() const;
    void accept(Visitor &) const;
private:
    std::string attr1_name;
    std::string attr2_name;
    Cmp cmp;
};

class ConstAttrExpressionFormula : public Formula {
public:
    ConstAttrExpressionFormula(std::string, int, Cmp);
    std::string get_attr_name() const;
    int get_const_value() const;
    Cmp get_cmp() const;
    void accept(Visitor &) const;
private:
    std::string attr_name;
    int const_value;
    Cmp cmp;
};

#endif