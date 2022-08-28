/*!
 *@author   chenghua.wang
 *@file     vectorization/veced_op.hpp
 *@brief    User should include this file if they want to use SSE or
 * Other CPU based methods to accelerate the calculation period.
 */

#ifndef _MGLORIA___VECED_OP_HPP_
#define _MGLORIA___VECED_OP_HPP_

#include "tensor.hpp"
#pragma once

#include "../depends.hpp"
#if MGLORIA_USE_SSE == 1
#include "./vectorization/__vec_sse.hpp"
#else
#include "./vectorization/__vec_none.hpp"
#endif  //  MGLORIA_USE_SSE == 1

namespace mgloria {
namespace expr {

/*!
 *@brief
 */
template<typename ExpressionType, typename DataType, vectorization::VecArch Arch>
struct VectorizedJob {
  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const;
  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType, Arch> EvalVec(index_t y, index_t x) const;
};

/*!
 *@brief
 */
template<typename Device, typename DataType, vectorization::VecArch Arch>
struct VectorizedJob<Tensor<Device, 1, DataType>, DataType, Arch> {
  explicit VectorizedJob(const Tensor<Device, 1, DataType>& t) : __data_ptr(t.__data_ptr) {}

  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const { return __data_ptr[x]; }
  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType, Arch> EvalVec(index_t y, index_t x) const {
    return vectorization::Vectorized<DataType, Arch>::Load(&__data_ptr[x]);
  }

 private:
  DataType* __data_ptr;
};

/*!
 *@brief
 */
template<typename Device, int Dims, typename DataType, vectorization::VecArch Arch>
struct VectorizedJob<Tensor<Device, Dims, DataType>, DataType, Arch> {
  explicit VectorizedJob(const Tensor<Device, Dims, DataType>& t)
      : __data_ptr(t.__data_ptr), m_Stride(t.m_Stride) {}

  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const {
    return __data_ptr[y * m_Stride + x];
  }
  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType, Arch> EvalVec(index_t y, index_t x) const {
    return vectorization::Vectorized<DataType, Arch>::Load(&__data_ptr[y * m_Stride + x]);
  }

 private:
  DataType* __data_ptr;
  index_t m_Stride;
};

}  // namespace expr
}  // namespace mgloria

#endif