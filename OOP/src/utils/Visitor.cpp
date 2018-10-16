#include "Visitor.hpp"

std::set<std::string> AttrVisitor::attributes(const Formula &f) {
    visit(f);
    return res;
}

void AttrVisitor::visit(const Formula &f) {
    f.accept(*this);
}

void AttrVisitor::visit(const TrueFormula &f) {}

void AttrVisitor::visit(const ImplicationFormula &f) {
    for (auto i = f.begin(); i != f.end(); i++) {
        (*i)->accept(*this);
    }
}

void AttrVisitor::visit(const DisjunctionFormula &f) {
    for (auto i = f.begin(); i != f.end(); i++) {
        (*i)->accept(*this);   
    }
}

void AttrVisitor::visit(const ConjunctionFormula &f) {
    for (auto i = f.begin(); i != f.end(); i++) {
        (*i)->accept(*this);   
    }
}

void AttrVisitor::visit(const NegationFormula &f) {
    f.formula().accept(*this);
}

void AttrVisitor::visit(const AttrAttrExpressionFormula &f) {
    res.insert(f.get_attr1_name());
    res.insert(f.get_attr2_name());
}

void AttrVisitor::visit(const ConstAttrExpressionFormula &f) {
    res.insert(f.get_attr_name());
}

bool EvaluateVisitor::value(const Formula &f, std::map<std::string, int> values) {
    this->values = values;
    res = 0;
    visit(f);
    return res;
}

void EvaluateVisitor::visit(const Formula &f) {
    f.accept(*this);
}

void EvaluateVisitor::visit(const TrueFormula &f) {
    res = true;
}

void EvaluateVisitor::visit(const ImplicationFormula &f) {
    bool tmp = 0;
    for (auto k : f) { 
        k->accept(*this);
        tmp = !(tmp) || res;
    }
    res = tmp;
}

void EvaluateVisitor::visit(const DisjunctionFormula &f) {
    bool tmp = 0;
    for (auto k : f) { 
        k->accept(*this);
        tmp = tmp || res;
    }
    res = tmp;
}

void EvaluateVisitor::visit(const ConjunctionFormula &f) {
    bool tmp = 1;
    for (auto k : f) { 
        k->accept(*this);
        tmp = tmp && res;
    }
    res = tmp;
}

void EvaluateVisitor::visit(const NegationFormula &f)  {
    f.formula().accept(*this);
    res = !res;
}

void EvaluateVisitor::visit(const AttrAttrExpressionFormula &f) {
    switch(f.get_cmp())
    {
        case Formula::G : {
            res = (values[f.get_attr1_name()] > values[f.get_attr2_name()]);
            return;
        }
        case Formula::GE : {
            res = (values[f.get_attr1_name()] >= values[f.get_attr2_name()]);
            return;
        }
        case Formula::L : {
            res = (values[f.get_attr1_name()] < values[f.get_attr2_name()]);
            return;
        }
        case Formula::LE : {
            res = (values[f.get_attr1_name()] <= values[f.get_attr2_name()]);
            return;
        }
        case Formula::E : {
            res = (values[f.get_attr1_name()] == values[f.get_attr2_name()]);
            return;
        }
        case Formula::NE : {
            res = (values[f.get_attr1_name()] != values[f.get_attr2_name()]);
        }
    }
}

void EvaluateVisitor::visit(const ConstAttrExpressionFormula &f) {
    switch(f.get_cmp())
    {
        case Formula::G : {
            res = (values[f.get_attr_name()] > f.get_const_value());
            return;
        }
        case Formula::GE : {
            res = (values[f.get_attr_name()] >= f.get_const_value());
            return;
        }
        case Formula::L : {
            res = (values[f.get_attr_name()] < f.get_const_value());
            return;
        }
        case Formula::LE : {
            res = (values[f.get_attr_name()] <= f.get_const_value());
            return;
        }
        case Formula::E : {
            res = (values[f.get_attr_name()] == f.get_const_value());
            return;
        }
        case Formula::NE : {
            res = (values[f.get_attr_name()] != f.get_const_value());
            return;
        }
    }
}


std::string StringGenVisitor::string_gen(const Formula &f) {
    std::string tmp = "";
    visit(f);
    if (res != "") {
        tmp += "[ ";
        tmp += res;
        tmp += " ] => ";
        return tmp;
    }
    return res;
}

void StringGenVisitor::visit(const Formula &f) {
    f.accept(*this);
}

void StringGenVisitor::visit(const TrueFormula &) {}

void StringGenVisitor::visit(const ImplicationFormula &f) {
    std::string tmp = "( ";
    int flag = 0;
    for (auto k : f) { 
        if (flag == 0) {
            k->accept(*this);
            tmp += res;
        }
        else {
            k->accept(*this);
            tmp += " -> ";
            tmp += res;
        }
        flag = 1;
    }
    tmp += " )";
    res = tmp;
}

void StringGenVisitor::visit(const DisjunctionFormula &f) {
std::string tmp = "( ";
    int flag = 0;
    for (auto k : f) { 
        if (flag == 0) {
            k->accept(*this);
            tmp += res;
        }
        else {
            k->accept(*this);
            tmp += " | ";
            tmp += res;
        }
        flag = 1;
    }
    tmp += " )";
    res = tmp;
}

void StringGenVisitor::visit(const ConjunctionFormula &f) {
    std::string tmp = "( ";
    int flag = 0;
    for (auto k : f) { 
        if (flag == 0) {
            k->accept(*this);
            tmp += res;
        }
        else {
            k->accept(*this);
            tmp += " & ";
            tmp += res;
        }
        flag = 1;
    }
    tmp += " )";
    res = tmp;
}

void StringGenVisitor::visit(const NegationFormula &f) {
    std::string tmp = "( ! ";
    f.formula().accept(*this);
    tmp += res;
    tmp += " )";
    res = tmp;
}

void StringGenVisitor::visit(const AttrAttrExpressionFormula &f) {
    std::string tmp = "( $ ";
    tmp += f.get_attr1_name();
    switch(f.get_cmp())
    {
        case Formula::G : {
            tmp += " > ";
            break;
        }
        case Formula::GE : {
            tmp += " >= ";
            break;
        }
        case Formula::L : {
            tmp += " < ";
            break;
        }
        case Formula::LE : {
            tmp += " <= ";
            break;
        }
        case Formula::E : {
            tmp += " == ";
            break;
        }
        case Formula::NE : {
            tmp += " != ";
            break;
        }
    }
    tmp += "$ ";
    tmp += f.get_attr2_name();
    tmp += " )";
    res = tmp;

}

void StringGenVisitor::visit(const ConstAttrExpressionFormula &f) {
    std::string tmp = "( $ ";
    tmp += f.get_attr_name();
    switch(f.get_cmp())
    {
        case Formula::G : {
            tmp += " > ";
            break;
        }
        case Formula::GE : {
            tmp += " >= ";
            break;
        }
        case Formula::L : {
            tmp += " < ";
            break;
        }
        case Formula::LE : {
            tmp += " <= ";
            break;
        }
        case Formula::E : {
            tmp += " == ";
            break;
        }
        case Formula::NE : {
            tmp += " != ";
            break;
        }
    }
    tmp += std::to_string(f.get_const_value());
    tmp += " )";
    res = tmp;
}

