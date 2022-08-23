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
#endif

// defined the const exp
#define MGLORIA_CONSTEXPR constexpr

// log system used is Google log
#define MGLORIA_GLOG 1

/*!
 *@brief    To align the data.
 *@details  x normally set to 4, 8, 16. This attribute will make
 * data aligned to x.
 * For example, you can use this macro like this:
 * float foo_1[10] MGLORIA_ALIGNED(4);
 * int32_t foo_2[10] MGLORIA_ALIGNED(8);
 */
#define MGLORIA_ALIGNED(x) __attribute__((aligned(x)))
