project(mgloria_test C CXX)

include_directories(${mgloria_head_dir})

set(file_list main-test.cpp)

option(TEST_TENSOR_SHAPE on "")

if (TEST_TENSOR_SHAPE)
list(APPEND file_list ./tensor/shape_test.hpp)
endif()

add_executable(mgloria_test
    ${file_list}
)