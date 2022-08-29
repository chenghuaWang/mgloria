// main-test must include all head files, in order to make
// the compile_command.json for clangd enable.
#define TEST_TENSOR_SHAPE 0
#define TEST_TENSOR_BASIC_OP 1

#if TEST_TENSOR_SHAPE == 1
#include "tensor/shape_test.hpp"
#endif
#if TEST_TENSOR_BASIC_OP == 1
#include "tensor/basic_op_test.hpp"
#endif

int main() {
  LOG << "Starting tests for MGloria project.\n";
#if TEST_TENSOR_SHAPE == 1
  __test_tensor_shape__();
#endif
#if TEST_TENSOR_BASIC_OP == 1
  __test_tensor_basic_OP__();
#endif
  return 0;
}