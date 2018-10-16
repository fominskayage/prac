#include "Formula.hpp"
#include "Visitor.hpp"




void ImplicationFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void DisjunctionFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void ConjunctionFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void NegationFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void AttrAttrExpressionFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void ConstAttrExpressionFormula::accept(Visitor &v) const {
    v.visit(*this);
}

void TrueFormula::accept(Visitor &v) const {
    v.visit(*this);
}


ConstAttrExpressionFormula::ConstAttrExpressionFormula(std::string str, int val, Formula::Cmp cmp) 
        : attr_name(str), const_value(val), cmp(cmp) {}

std::string ConstAttrExpressionFormula::get_attr_name() const {
    return attr_name;
}

int ConstAttrExpressionFormula::get_const_value() const {
    return const_value;
}

Formula::Cmp ConstAttrExpressionFormula::get_cmp() const {
    return cmp;
}

AttrAttrExpressionFormula::AttrAttrExpressionFormula(std::string first, std::string second, Formula::Cmp cmp) :
        attr1_name(first), attr2_name(second), cmp(cmp) {}

std::string AttrAttrExpressionFormula::get_attr1_name() const {
    return attr1_name;
}
    

std::string AttrAttrExpressionFormula::get_attr2_name() const {
    return attr2_name;
}


Formula::Cmp AttrAttrExpressionFormula::get_cmp() const {
    return cmp;
}

