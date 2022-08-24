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

#include <ostream>
#if _MSC_VER > 1000
#pragma once
#error "Detected you are using MSVC to compile. MSVC is not support. Also, windows platform."
#endif

// include basic files.
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <climits>
#include <sstream>
#include <string>

// Flags for User to set.
#ifndef MGLORIA_USE_CUDA
#define MGLORIA_USE_CUDA 1
#endif

#ifndef MGLORIA_USE_CUDNN
#define MGLORIA_USE_CUDNN 0
#endif

#ifndef MGLORIA_USE_MKL
#define MGLORIA_USE_MKL 1
#endif

#ifndef MGLORIA_USE_SSE
#define MGLORIA_USE_SSE 1
#endif

// include files for CUDA and C-Blas
#if MGLORIA_USE_MKL
#include <mkl/mkl_blas.h>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_vsl.h>
#include <mkl/mkl_vsl_functions.h>
#include <mkl/mkl_version.h>
#endif
#if MGLORIA_USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#if MGLORIA_USE_CUDNN
// TODO include something.
#endif

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
        std::cout << "[Warn] [" << file << " at line " << line << "] ";
        return std::cout;
      }
      case _log_type::err: {
        std::cerr << "[Err] [" << file << " at line " << line << "] ";
        return std::cerr;
      }
      case _log_type::debug: {
        std::cout << "[Debug] [" << file << " at line " << line << "] ";
        return std::cout;
      }
      default: {
        std::cout << "[Info] [" << file << " at line " << line << "] ";
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

/*!
 *@brief CHECK if the condition is True.
 */
#define LOG_CHECK(x) \
  if (!(x)) { LOG_ERR << "Check [" << #x << "] Fail.\n" }
#define CHECK_LOWER_THAN(x, y) LOG_CHECK((x) < (y))
#define CHECK_GREATER_THAN(x, y) LOG_CHECK((x) > (y))
#define CHECK_EQUAL(x, y) LOG_CHECK((x) == (y))
#define CHECK_LOWER_EQUAL(x, y) LOG_CHECK((x) <= (y))
#define CHECK_GREATER_EQUAL(x, y) LOG_CHECK((x) >= (y))
#define CHECK_NOT_EQUAL(x, y) LOG_CHECK((x) != (y))
#define CHECK_NULL(x) LOG_CHECK((x) != NULL)

/*!
 *@brief Define the guard macro for CUDA Call. Also Error throw.
 */
#if MGLORIA_USE_CUDA == 1

#define GUARD_CUDA_CALL(func)                                           \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e == cudaErrorCudartUnloading) { throw cudaGetErrorString(e); } \
    LOG_CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e);   \
  }

#endif  // MGLORIA_USE_CUDA == 1

#define MGLORIA_ARRAY_BOUND_CHECK 0

#endif  // MGLORIA_PREPARE_HPP_
