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

  float data_a[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  float data_b[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  float data_c[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  Tensor<CPU, 3> A(data_a, makeShape3d(2, 2, 2));
  Tensor<CPU, 3> B(data_b, makeShape3d(2, 2, 2));
  Tensor<CPU, 3> C(data_c, makeShape3d(2, 2, 2));
  A = B + C;
  A = expr::Func<ReLU>(A);
  std::cout << A;
  ShutdownTensorComputeMachine<CPU>(0);
}