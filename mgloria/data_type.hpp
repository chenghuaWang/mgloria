/*!
 *@author   chenghua.wang
 *@file     data_type.hpp
 *@brief    Define the element datatype of tensor.
 *@note     Half is not support. Float, Double, Int32, Int64, Int8 is OK.
 */

#ifndef _MGLORIA_DATA_TYPE_HPP
#define _MGLORIA_DATA_TYPE_HPP

#include "prepare.hpp"
namespace mgloria {
typedef int32_t index_t;
typedef int64_t index_long_t;
typedef int32_t openmp_index_t;
typedef float default_t;

enum class TypeType : uint8_t { Float32 = 0, Float64, Int32, Int64, Int8, UInt8 };

template<typename DataType>
struct ElementType;

template<>
struct ElementType<float> {
  static const TypeType Type = TypeType::Float32;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_32F;
#endif
};

template<>
struct ElementType<double> {
  static const TypeType Type = TypeType::Float64;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_64F;
#endif
};

template<>
struct ElementType<int32_t> {
  static const TypeType Type = TypeType::Int32;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_32I;
#endif
};

template<>
struct ElementType<int64_t> {
  static const TypeType Type = TypeType::Int64;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_64I;
#endif
};

template<>
struct ElementType<int8_t> {
  static const TypeType Type = TypeType::Int8;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_8I;
#endif
};

template<>
struct ElementType<uint8_t> {
  static const TypeType Type = TypeType::UInt8;
#if MGLORIA_USE_CUDA
  static const cudaDataType_t CUDAType = CUDA_R_8U;
#endif
};

}  // namespace mgloria

#endif