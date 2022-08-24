// main-test must include all head files, in order to make
// the compile_command.json for clangd enable.
#define TEST_TENSOR_SHAPE 1

#if TEST_TENSOR_SHAPE == 1
#include "tensor/shape_test.hpp"
#endif

int main() {
  LOG << "Starting tests for MGloria project.\n";
  __test_tensor_shape__();
  return 0;
}