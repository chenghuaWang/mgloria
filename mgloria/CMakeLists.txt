project(
    mgloria
    VERSION 1.0
    DESCRIPTION "The matrix and parameter server for gloria framework"
    LANGUAGES C CXX
)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

file(GLOB_RECURSE mgloria_file_hpps *.hpp)
file(GLOB_RECURSE mgloria_file_cuda_hpps *.cuh)

add_library(mgloria INTERFACE
    ${mgloria_file_hpps}
    ${mgloria_file_cuda_hpps}
)
target_include_directories(mgloria INTERFACE ./)
target_compile_features(mgloria INTERFACE cxx_std_20)