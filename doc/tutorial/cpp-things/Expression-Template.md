Expression Template
===

The `Expression Template` is actually a application of `Curiously Recurring Template Pattern(CRTP)`. And it is a kind of trick of cpp template programming. For example, if I want to calculate `int A = B + C + D`. I will first get the result of `tmp = B + C`, and then `tmp` will sum to `D` to get the final answer. But, it will become complex if `A, B, C` is tensor. The parameter `tmp` will use lots of memory if tensor is large or the compute graph is deep and large. Also, the separate calculating of `A + B + C` will bring more loops which make it time consuming.

I assumed, all of the tensor I used bellow is $8 \times 8$ or $8 \times 8 \times 8$

## The Naive way.

pseudo code

```cpp
// Naive way to calculate A + B + C
tensor<int, 8, 8> A, B, C;
tensor<int, 8, 8> tmp, ans;
for (uint8_t i = 0; i < 8; ++i){
    for (uint8_t j = 0; j < 8; ++j) {
        tmp[i][j] = A[i][j] + B[i][j];
    }
}

for (uint8_t i = 0; i < 8; ++i){
    for (uint8_t j = 0; j < 8; ++j) {
        ans[i][j] = tmp[i][j] + c[i][j];
    }
}
```

As you can see. The above pseudo code run 4 times loops and use 2 times of memory to calculate `A + B + C`. What I want it works like is the pseudo code below

```cpp
//Much better way to calculate.
tensor<int, 8, 8> A, B, C;
tensor<int, 8, 8> ans;
for (uint8_t i = 0; i < 8; ++i){
    for (uint8_t j = 0; j < 8; ++j) {
        ans[i][j] = A[i][j] + B[i][j] + C[i][j];
    }
}
```

To make the expression works like this, some abstract need to be implemented.

## Using Dynamic Polymorphism.

In order to make the expression work in a lazy way(Don't compute on each element immediately, but first record the expression, then calculate them in one time). Some Abstract need to be proposed. It works like the code below.

```cpp
struct Exp {
    virtual double eval(int i, int j, int k) const = 0;
    virtual ~Exp() = default;
};

struct Tensor : public Exp {
    // ...
    double eval(int i, int j, int k) const override {
        return (*this)[i, j, k];
    }
    Tensor operator=(const Exp& exp) {
        for(int i = 0; i < 8; ++i) 
            for(int j = 0; j < 8; ++j)
                for(int k = 0; k < 8; ++k)
                    (*this)[i, j, k] = exp.eval(i, j, k);
    }
};

struct AddExp : public Exp {
    Exp* loperand, roperand;
    AddExp(Exp* l, Exp* r) : loperand(l), roperand(r) {}
    double eval(int i, int j, int k) const override {
        return loperand->eval(i, j, k) + roperand->eval(i, j, k);
    }
};
```

This process looks ideal, with no temporary storage and just one loop. However, the virtual function is introduced, the function inlining is not guaranteed, and it has to bear the cost of looking up the virtual table, so the efficiency is not necessarily better than the implementation we mentioned at the beginning.

## Using template.

The core of CRTP is to use the derived class as the template parameter of the base class template, so that the base class template records the information of the derived class. The dynamic polymorphism implemented by virtual functions before is to record the polymorphism information at runtime through the virtual table of the derived class; while the static polymorphism implemented by CRTP is to record the information of the derived class at compile time through the template parameters of the base class.

```cpp
template<typename SubType>
struct Exp {
    // significant
    SubType* self(void) {
        return static_cast<SubType*>(this);
    };
    double eval(int i, int j, int k) const {
        return self()->eval(i, j, k);
    }
};

struct Tensor : public Exp<Tensor> {
    //...
    double eval(int i, int j, int k) const {
        return (*this)[i, j, k];
    }

    template<typename SubType>
    Tensor operator=(const Exp<SubType>& exp) {
        for(int i = 0; i < 8; ++i) 
            for(int j = 0; j < 8; ++j)
                for(int k = 0; k < 8; ++k)
                    (*this)[i, j, k] = exp.eval(i, j, k);
    }
};

template<typename LhsType, typename RhsType>
struct AddExp : public Exp<AddExp<LhsType, RhsType>> {
    Exp<LhsType>* loperand;
    Exp<RhsType>* roperand;
    AddExp(Exp<LhsType>* l, Exp<RhsType>* r) 
            : loperand(l), roperand(r) {}
    double eval(int i, int j, int k) const {
        return loperand->eval(i, j, k) + roperand->eval(i, j, k);
    }
};
```

The current implementation does not have any virtual functions. All eval() calls are compiled into the eval methods of the derived classes we need at compile time. Function inlining can be reliably guaranteed, which basically meets our expectations.

## references
[1] https://en.wikipedia.org/wiki/Expression_templates