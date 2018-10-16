#ifndef PRAC_INCLUDE_UTILS_VISITOR_HPP
#define PRAC_INCLUDE_UTILS_VISITOR_HPP

#include "Formula.hpp"
#include <map>
#include <set>
#include <string>

class Visitor {
public:
    virtual void visit(const Formula &) = 0;
    virtual void visit(const TrueFormula &) = 0;
    virtual void visit(const ImplicationFormula &) = 0;
    virtual void visit(const DisjunctionFormula &) = 0;
    virtual void visit(const ConjunctionFormula &) = 0;
    virtual void visit(const NegationFormula &) = 0;
    virtual void visit(const AttrAttrExpressionFormula &) = 0;
    virtual void visit(const ConstAttrExpressionFormula &) = 0;
    virtual ~Visitor() = default;
};


class AttrVisitor : public Visitor
{
public:
    std::set<std::string> attributes(const Formula &);
    void visit(const Formula &);
    void visit(const TrueFormula &);
    void visit(const ImplicationFormula &);
    void visit(const DisjunctionFormula &);
    void visit(const ConjunctionFormula &);
    void visit(const NegationFormula &);
    void visit(const AttrAttrExpressionFormula &);
    void visit(const ConstAttrExpressionFormula &);
private:
    std::set<std::string> res;
};

class EvaluateVisitor : public Visitor
{
public:
    bool value(const Formula &, std::map<std::string, int>);
    void visit(const Formula &);
    void visit(const TrueFormula &);
    void visit(const ImplicationFormula &);
    void visit(const DisjunctionFormula &);
    void visit(const ConjunctionFormula &);
    void visit(const NegationFormula &);
    void visit(const AttrAttrExpressionFormula &);
    void visit(const ConstAttrExpressionFormula &);
private:
    std::map<std::string, int> values;
    bool res;
};

class StringGenVisitor : public Visitor
{
public:
    std::string string_gen(const Formula &);
    void visit(const Formula &);
    void visit(const TrueFormula &);
    void visit(const ImplicationFormula &);
    void visit(const DisjunctionFormula &);
    void visit(const ConjunctionFormula &);
    void visit(const NegationFormula &);
    void visit(const AttrAttrExpressionFormula &);
    void visit(const ConstAttrExpressionFormula &);
private:
    std::string res;
};
#endif
