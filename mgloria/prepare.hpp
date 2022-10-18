/*!
 *@author   chenghua.wang
 *@file     prepare.hpp
 *@brief    basic settings. But simplified for implementation.
 *@note     Note that this file is just for Linux/Unix system.
 * Just Tested on Ubuntu20.04, and assumed only use CUDA for GPU
 * and MKL for cpu. You can only choose which device you want to
 * run the program, but the backend can not be changed.
 */

#ifndef MGLORIA_PREPARE_HPP
#define MGLORIA_PREPARE_HPP

///! All files in this head only lib using #pragma once. Make sure your compiler support it.
#pragma once

#if _MSC_VER
#error "Detected you are using MSVC to compile. MSVC is not support. Also, windows platform."
#endif

#ifdef __APPLE__
#error "Detected you are using MacOS, Mac not support MKL and CUDA. Compile interrupt."
#endif  // __APPLE__

// #if __clang__
// #include <__clang_cuda_runtime_wrapper.h>
// #endif

// include basic files.
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <climits>
#include <sstream>
#include <string>
#include <limits>
#include <cstdint>
#include <ostream>

// Flags for User to set.
#ifndef MGLORIA_USE_CUDA
#define MGLORIA_USE_CUDA 1
#endif

#ifndef MGLORIA_USE_CUDNN
#define MGLORIA_USE_CUDNN 0
#endif

#ifndef MGLORIA_USE_MKL
#define MGLORIA_USE_MKL 0
#endif

#ifndef MGLORIA_USE_BLAS
#define MGLORIA_USE_BLAS 1
#endif

#ifndef MGLORIA_USE_SSE
#define MGLORIA_USE_SSE 1
#endif

#define MGLORIA_ARRAY_BOUND_CHECK 0
#define MGLORIA_CHECK_NULL_MEM_PTR 1
#define MGLORIA_PAD_TO_ALIGN 1
#define MGLORIA_RUNTIME_SHAPE_CHECK 1
#define MGLORIA_RUNTIME_DEVICE_TYPE_CHECK 1
#define MGLORIA_MAX_SHOW_LENGTH 8

#if MGLORIA_USE_SSE == 1
#define MGLORIA_VECTORIZATION_ARCH ::mgloria::vectorization::VecArch::SSE_Arch
#else
#define MGLORIA_VECTORIZATION_ARCH ::mgloria::vectorization::VecArch::NONE_Arch
#endif
#define MGLORIA_DEFAULT_ALIGNBYTES 4

// include files for CUDA and C-Blas
#if MGLORIA_USE_MKL
#include <mkl/mkl_blas.h>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_vsl.h>
#include <mkl/mkl_vsl_functions.h>
#include <mkl/mkl_version.h>
#endif
#if MGLORIA_USE_BLAS
#include <cblas.h>
#endif
#if MGLORIA_USE_CUDA == 1
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#if MGLORIA_USE_CUDNN
// TODO include something.
#endif

#define MGLORIA_VECTORIZATION_TRUE true
#define MGLORIA_VECTORIZATION_FALSE false

// inline symbol for template code used in both gpu and cpu
#ifdef MGLORIA_INLINE
#error "You can not predefine MGLORIA_INLINE. It's used in this lib."
#endif
#define MGLORIA_FORCE_INLINE inline __attribute__((always_inline))
#ifdef __CUDACC__
#define MGLORIA_INLINE_XPU MGLORIA_FORCE_INLINE __device__ __host__
#else
#define MGLORIA_INLINE_XPU MGLORIA_FORCE_INLINE
#endif
// inline symbol for template code used just on cpu
#define MGLORIA_INLINE_CPU MGLORIA_FORCE_INLINE
#define MGLORIA_INLINE_NORMAL inline

// defined the const exp
#define MGLORIA_CONSTEXPR constexpr

// log system used is Google log
#define MGLORIA_GLOG 0

/*!
 *@brief    To align the data.
 *@details  x normally set to 4, 8, 16. This attribute will make
 * data aligned to x.
 * For example, you can use this macro like this:
 * float foo_1[10] MGLORIA_ALIGNED(4);
 * int32_t foo_2[10] MGLORIA_ALIGNED(8);
 */
#define MGLORIA_ALIGNED(x) __attribute__((aligned(x)))

/*!
 *@brief    A LOG system. Logging to terminal only.
 */
#if MGLORIA_GLOG == 0
#include <iostream>
enum class _log_type { all = 0, debug, warn, info, err };

class _log_stream {
 public:
  static std::ostream& make_stream(const char* file, int line, const _log_type& t) {
    switch (t) {
      case _log_type::warn: {
        std::cout << "[Warn] [" << file << "(" << line << ", 0)"
                  << "] ";
        return std::cout;
      }
      case _log_type::err: {
        std::cerr << "[Err] [" << file << "(" << line << ", 0)"
                  << "] ";
        return std::cerr;
      }
      case _log_type::debug: {
        std::cout << "[Debug] [" << file << "(" << line << ", 0)"
                  << "] ";
        return std::cout;
      }
      default: {
        std::cout << "[Info] [" << file << "(" << line << ", 0)"
                  << "] ";
        return std::cout;
      }
    }
  }
};

#endif  // MGLORIA_GLOG == 0

/*!
 *@brief The LOG macro for easily use.
 */
#define LOG_INFO _log_stream::make_stream(__FILE__, __LINE__, _log_type::info)
#define LOG_ERR _log_stream::make_stream(__FILE__, __LINE__, _log_type::err)
#define LOG_WARN _log_stream::make_stream(__FILE__, __LINE__, _log_type::warn)
#define LOG_DEBUG _log_stream::make_stream(__FILE__, __LINE__, _log_type::debug)
#define LOG _log_stream::make_stream(__FILE__, __LINE__, _log_type::all)

template<typename T>
MGLORIA_INLINE_NORMAL std::ostream &operator,(std::ostream&out, const T&t) {
  out << t;
  return out;
}

MGLORIA_INLINE_NORMAL std::ostream &operator,(std::ostream&out, std::ostream&(*f)(std::ostream&)) {
  out << f;
  return out;
}

/*!
 *@brief CHECK if the condition is True.
 */
#define LOG_CHECK(x, ...)                            \
  if (!(x)) {                                        \
    LOG_ERR << "Failed when testing " << #x << "\n"; \
    std::cerr, __VA_ARGS__, std::endl;               \
    std::exit(1);                                    \
  }

#define CHECK_LOWER_THAN(x, y, ...) LOG_CHECK((x) < (y), __VA_ARGS__)
#define CHECK_GREATER_THAN(x, y, ...) LOG_CHECK((x) > (y), __VA_ARGS__)
#define CHECK_EQUAL(x, y, ...) LOG_CHECK((x) == (y), __VA_ARGS__)
#define CHECK_LOWER_EQUAL(x, y, ...) LOG_CHECK((x) <= (y), __VA_ARGS__)
#define CHECK_GREATER_EQUAL(x, y, ...) LOG_CHECK((x) >= (y), __VA_ARGS__)
#define CHECK_NOT_EQUAL(x, y, ...) LOG_CHECK((x) != (y), __VA_ARGS__)
#define CHECK_NULL(x, ...) LOG_CHECK((x) != NULL, __VA_ARGS__)

/*!
 *@brief Define the guard macro for CUDA Call. Also Error throw.
 */
#if MGLORIA_USE_CUDA == 1

#define GUARD_CUDA_CALL(func)                                           \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e == cudaErrorCudartUnloading) { throw cudaGetErrorString(e); } \
    LOG_CHECK(e == cudaSuccess, "CUDA Failed");                         \
  }

#endif  // MGLORIA_USE_CUDA == 1

template<typename DataType>
MGLORIA_INLINE_XPU DataType MINLimit();

template<>
MGLORIA_INLINE_XPU float MINLimit<float>() {
  return std::numeric_limits<float>().min();
}

template<>
MGLORIA_INLINE_XPU double MINLimit<double>() {
  return std::numeric_limits<double>().min();
}

template<>
MGLORIA_INLINE_XPU int32_t MINLimit<int32_t>() {
  return std::numeric_limits<int32_t>().min();
}

template<>
MGLORIA_INLINE_XPU int64_t MINLimit<int64_t>() {
  return std::numeric_limits<int64_t>().min();
}

template<>
MGLORIA_INLINE_XPU int8_t MINLimit<int8_t>() {
  return std::numeric_limits<int8_t>().min();
}

template<>
MGLORIA_INLINE_XPU uint8_t MINLimit<uint8_t>() {
  return std::numeric_limits<uint8_t>().min();
}

template<typename DataType>
MGLORIA_INLINE_XPU DataType MAXLimit();

template<>
MGLORIA_INLINE_XPU float MAXLimit<float>() {
  return std::numeric_limits<float>().max();
}

template<>
MGLORIA_INLINE_XPU double MAXLimit<double>() {
  return std::numeric_limits<double>().max();
}

template<>
MGLORIA_INLINE_XPU int32_t MAXLimit<int32_t>() {
  return std::numeric_limits<int32_t>().max();
}

template<>
MGLORIA_INLINE_XPU int64_t MAXLimit<int64_t>() {
  return std::numeric_limits<int64_t>().max();
}

template<>
MGLORIA_INLINE_XPU int8_t MAXLimit<int8_t>() {
  return std::numeric_limits<int8_t>().max();
}

template<>
MGLORIA_INLINE_XPU uint8_t MAXLimit<uint8_t>() {
  return std::numeric_limits<uint8_t>().max();
}

#define MGLORIA_IS_NAN(func) std::isnan(func)
#define MGLORIA_IS_INF(func) std::isinf(func)

#define NO_TYPE_PTR void*

#define MGLORIA_SHAPE_ERROR_EXIT 1
#define MGLORIA_TYPE_ERROR_EXIT 2
#define MGLORIA_MEM_ERROR_EXIT 3
#define MGLORIA_OTHER_ERROR_EXIT 4

#endif  // MGLORIA_PREPARE_HPP_
