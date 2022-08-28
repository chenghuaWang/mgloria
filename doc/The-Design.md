The Arch and Design of MGloria
===
This file can help you figure out how is MGloria organized, and how it works.

In order to understand the arch of this head only lib, some vital abstractions need to be introduced first. As you may know, the MGloria is highly depend on the Expression template feature(You can find details in [Expression-template](/doc/tutorial/cpp-things/Expression-Template.md)). All data, such as tensor is inherited from the basic expression template class. And for lazy compute, all formula will finally being translated to a huge expression Tree. This Tree is not build explicit, it's built by the complier when phase the template class in the compile time implicitly. After we got those expression trees, we need to execute them, and `Job` abstraction is born for it. `Job` abstraction is implemented as a template class. Each sub expression is corresponding to one of `Job`, if you dive into the codebase, you will find that, `Job` is generated using expression tree recursively. All actual evaluate function is implemented in `Job`, and both CPU/GPU method is provided. When the expression tree meet an `=`, it will call the expression dispatcher to evaluate the expression using multi `Job`s automatically. 

## Expression abstraction.

## Tensor abstraction.

## `Job` abstraction.

## How to add a new OP ?

## A little example of how a formula is interpreted and evaluated.

