#include "core.hpp"

inline void __test_tensor_basic_OP__() {
  using namespace mgloria;

  InitTensorComputeMachine<CPU>(0);

  struct ReLU {
    MGLORIA_INLINE_NORMAL static float Do(float t) {
      if (t > 0.f) return t;
      return 0.f;
    }
  };

  auto __stream__ = NewStream<CPU>(0);

  Tensor<CPU, 3> A = NewTensor(makeShape3d(2, 3, 5), true, 2.f, true, __stream__);
  Tensor<CPU, 3> B = NewTensor(makeShape3d(2, 3, 5), true, 1.f, true, __stream__);
  Tensor<CPU, 3> C = NewTensor(makeShape3d(2, 3, 5), true, 5.f, true, __stream__);

  A = B + C;
  // A = expr::implicit_dot(B, C);
  A = expr::Func<ReLU>(A);

  std::cout << A;
  std::cout << B;
  std::cout << C;
  FreeStream(__stream__);
  ShutdownTensorComputeMachine<CPU>(0);
}
