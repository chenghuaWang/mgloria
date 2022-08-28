Trivial
===

## Clangd
The Clangd doesn't work if some cuda args is not pass to clang compiler. The clang++12, for now, it seems just support the cuda vision lower than 10.0. If you want Clangd can recognize `__device__, __host__, __CUDACC__`, you should add those codes in your main CMake files.

```CMake
if (USE_CUDA)
add_definitions("-xcuda")
add_definitions("--cuda-path=/usr/local/cuda-11.7")
add_definitions("--cuda-gpu-arch=sm_80")
endif()
```

where `-xcuda` is necessary. And if you have multi vision of cuda, you should config the cudaToolKit path manually by setting the `--cuda-path` to the SDK you want. Also, Clang++ will check your GPU's Arch. And it will throw error if the cuda vision mismatch the Arch. For A100 sm_80 is OK. RT30XX is sm_86, while clang12 seems not support such high vision Arch.

## C++20
The c++ volatile which I heavily depends on has changed. According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1152r1.html. I don't want to adapt this new law anymore..:-( . The c++ standard I used is back to c++11, it also works well.
