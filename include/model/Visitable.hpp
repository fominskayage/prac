#ifndef PRAC_INCLUDE_MODEL_VISITABLE_HPP
#define PRAC_INCLUDE_MODEL_VISITABLE_HPP

class Visitor;

class Visitable {
public:
    virtual void accept(Visitor &) const = 0;
    virtual ~Visitable() = default;
};

#endif
