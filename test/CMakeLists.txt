project(mgloria_test C CXX CUDA)

include_directories(${mgloria_head_dir})

set(file_list main-test.cpp)

option(TEST_TENSOR_SHAPE off "")
option(TEST_TENSOR_BASIC_OP on "")

if (TEST_TENSOR_SHAPE)
list(APPEND file_list ./tensor/shape_test.hpp)
endif()
if (TEST_TENSOR_BASIC_OP)
list(APPEND file_list ./tensor/basic_op_test.hpp)
endif()

add_executable(mgloria_test
    ${file_list}
)