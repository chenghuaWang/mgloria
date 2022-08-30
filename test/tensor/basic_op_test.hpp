#include "core.hpp"

inline void __test_tensor_basic_OP__() {
  using namespace mgloria;

  InitTensorComputeMachine<CPU>(0);

  struct ReLU {
    MGLORIA_INLINE_NORMAL static float Do(float t) { return t; }
  };

  auto __stream__ = NewStream<CPU>(0);

  Tensor<CPU, 3> A = NewTensor(makeShape3d(4, 4, 4), true, 1.f, true, __stream__);
  Tensor<CPU, 3> B = NewTensor(makeShape3d(4, 4, 4), true, 1.f, true, __stream__);
  Tensor<CPU, 3> C = NewTensor(makeShape3d(4, 4, 4), true, 1.f, true, __stream__);
  A = B + C;
  A = expr::Func<ReLU>(A);
  std::cout << A;
  ShutdownTensorComputeMachine<CPU>(0);
}