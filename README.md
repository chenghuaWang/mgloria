MGloria: Matrix and Parameter server for Gloria framework
======
This lib is for practicing purpose and still fighting with bugs

This mgloria is a head only lib focus on Matrix and parameter server. For Practicing purpose. This lib is heavily depends on the expression template. MKL is the backend of CPU-Tensor, and CUDA/CUDNN is for GPU-Tensor. Doxygen style comment is used for doc.

For example, if you want to compute this:
```c++
auto __stream__ = NewStream<CPU>(0);

Tensor<CPU, 3> A = NewTensor(makeShape3d(2, 3, 5), true, 2.f, true, __stream__);
Tensor<CPU, 3> B = NewTensor(makeShape3d(2, 3, 5), true, 1.f, true, __stream__);
Tensor<CPU, 3> C = NewTensor(makeShape3d(2, 3, 5), true, 5.f, true, __stream__);

A = B + C;
std::cout << A;
```

The Matrix engine will first interpret your formula to a expression tree. Then this tree will be passed to `Job` abstraction, the `Job` will finally decide which device should the operations run on, and what kind of optimization should be made.

## On the route
- [ ] [ ALL ] The Print function for Aligned Data is Buggy.
- [ ] [ ALL ] OpenMP for CPU-Tensor to use. Need Test and Config.
- [ ] [ OP ] Implicit GEMM For CPU-Tensor. Need to change and Optimize.
- [ ] [ OP ] Basic ops for neural net need to implemented.
- [ ] [ BUILD ] CMake files more robustness and Easily to use
- [ ] [ ALL ] Comment for all functions and template class.

- [ ] [ OP ] Implicit GEMM For GPU-Tensor. TensorCore involved in.