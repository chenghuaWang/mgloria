#include "core.hpp"

inline void __test_tensor_shape__() {
  using namespace mgloria;
  LOG << "-------- Starting test [Tensor][Shape] \n";
  Shape<1> shape1 = makeShape1d(8);
  Shape<2> shape2 = makeShape2d(8, 8);
  Shape<3> shape3 = makeShape3d(8, 8, 8);
  Shape<4> shape4 = makeShape4d(8, 8, 8, 8);
  Shape<5> shape5 = makeShape5d(8, 8, 8, 8, 8);
  LOG << shape1.str();
  LOG << shape2.str();
  LOG << shape3.str();
  LOG << shape4.str();
  LOG << shape5.str();
  LOG << "Flatten 1d " << shape5.Flatten1D().str();
  LOG << "Flatten 2d " << shape5.Flatten2D().str();
  LOG << "CUDA Shape " << shape5.CudaShape().str();
  LOG << "Slice " << shape5.Slice<1, 2>().str();
  LOG << "Size" << shape5.Size();
  LOG << "Sub Size " << shape5.SubSize(1, 2);
  LOG << "-------- Successfully tested [Tensor][Shape] \n";
}