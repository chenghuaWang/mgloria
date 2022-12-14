/*!
 *@author   chenghua.wang
 *@file     vectorization/veced_op.hpp
 *@brief    User should include this file if they want to use SSE or
 * Other CPU based methods to accelerate the calculation period.
 */

#ifndef _MGLORIA___VECED_OP_HPP_
#define _MGLORIA___VECED_OP_HPP_

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
      : __data_ptr(t.__data_ptr), m_Stride(t.m_Stride_) {}

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

/*!
 *@brief
 */
template<typename DataType, vectorization::VecArch Arch>
class VectorizedJob<ScalarExpr<DataType>, DataType, Arch> {
 public:
  explicit VectorizedJob(DataType scalar) : m_Scalar(scalar) {}

  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType, Arch> EvalVec(index_t y, index_t x) const {
    return vectorization::Vectorized<DataType, Arch>::Fill(m_Scalar);
  }
  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const { return m_Scalar; }

 private:
  DataType m_Scalar;
};

/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, exprType EType, typename DataType,
         vectorization::VecArch Arch>
class VectorizedJob<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType, Arch> {
 public:
  VectorizedJob(const VectorizedJob<A_T, DataType, Arch>& lhs,
                const VectorizedJob<B_T, DataType, Arch>& rhs)
      : m_lhs(lhs), m_rhs(rhs) {}
  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType, Arch> EvalVec(index_t y, index_t x) const {
    return vectorization::VectorizedOP<OP, DataType, Arch>::Do(m_lhs.EvalVec(y, x),
                                                               m_rhs.EvalVec(y, x));
  }
  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const {
    return OP::Do(m_lhs.Eval(y, x), m_rhs.Eval(y, x));
  }

 private:
  VectorizedJob<A_T, DataType, Arch> m_lhs;
  VectorizedJob<B_T, DataType, Arch> m_rhs;
};

/*!
 *@brief
 */
template<typename OP, typename A_T, int EType, typename DataType, vectorization::VecArch Arch>
class VectorizedJob<UnaryExpr<OP, A_T, DataType, EType>, DataType, Arch> {
 public:
  VectorizedJob(const VectorizedJob<A_T, DataType, Arch>& src) : m_src(src) {}
  MGLORIA_INLINE_CPU vectorization::Vectorized<DataType> EvalVec(index_t y, index_t x) const {
    return vectorization::VectorizedOP<OP, DataType, Arch>::Do(m_src.EvalVec(y, x));
  }
  MGLORIA_INLINE_CPU DataType Eval(index_t y, index_t x) const { return OP::Do(m_src.Eval(y, x)); }

 private:
  VectorizedJob<A_T, DataType, Arch> m_src;
};

/*!
 *@brief
 */
template<vectorization::VecArch Arch, typename DataType>
inline VectorizedJob<ScalarExpr<DataType>, DataType, Arch> NewVectorizedJob(
    const ScalarExpr<DataType>& e) {
  return VectorizedJob<ScalarExpr<DataType>, DataType, Arch>(e.scale_value);
}

/*!
 *@brief
 */
template<vectorization::VecArch Arch, typename T, typename DataType>
inline VectorizedJob<T, DataType, Arch> NewVectorizedJob(const RValueExpr<T, DataType>& e) {
  return VectorizedJob<T, DataType, Arch>(e.Self());
}

/*!
 *@brief
 */
template<vectorization::VecArch Arch, typename OP, typename A_T, typename DataType, exprType EType>
inline VectorizedJob<UnaryExpr<OP, A_T, DataType, EType>, DataType, Arch> NewVectorizedJob(
    const UnaryExpr<OP, A_T, DataType, EType>& e) {
  return VectorizedJob<UnaryExpr<OP, A_T, DataType, EType>, DataType, Arch>(
      NewVectorizedJob<Arch>(e.m_src));
}

/*!
 *@brief
 */
template<vectorization::VecArch Arch, typename OP, typename A_T, typename B_T, typename DataType,
         exprType EType>
inline VectorizedJob<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType, Arch> NewVectorizedJob(
    const BinaryExpr<OP, A_T, B_T, DataType, EType>& e) {
  return VectorizedJob<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType, Arch>(
      NewVectorizedJob<Arch>(e.m_lhs), NewVectorizedJob<Arch>(e.m_rhs));
}

/*!
 *@brief
 */
template<typename LeftValue, typename E, int Dims, typename DataType, vectorization::VecArch Arch>
MGLORIA_INLINE_NORMAL void ExecuteVectorizedJob(Tensor<CPU, Dims, DataType> _dst,
                                                const VectorizedJob<E, DataType, Arch>& plan) {
  Tensor<CPU, 2, DataType> dst = _dst.Flatten2D();
  const index_t xlen = vectorization::FloorAlign<Arch, DataType>(dst.size(1));
  const size_t vec_size = vectorization::Vectorized<DataType, Arch>::num;

  // LOG << "------------------- DEBUG LLVM ---------------------------------\n";
  // LOG << dst.m_Shape.str();
  // LOG << xlen << std::endl;
  // LOG << vec_size << std::endl;
  // LOG << "------------------- DEBUG LLVM ---------------------------------\n";

#ifndef __CUDACC__
#pragma omp parallel for
#endif
  for (openmp_index_t y = 0; y < dst.size(0); ++y) {
    // This pat is can be vectorized.
    for (index_t x = 0; x < xlen; x += vec_size) {
      vectorization::VectorizedSaver<LeftValue, DataType, Arch>::Do(&dst[y][x], plan.EvalVec(y, x));
    }
    // The left can not be vectorized.
    for (index_t x = xlen; x < dst.size(1); ++x) { LeftValue::Do(dst[y][x], plan.Eval(y, x)); }
  }
}
}  // namespace expr

// ############################### Below for Vectorization Enable check. #####################
template<typename E, vectorization::VecArch Arch>
struct VecCheck {
  static const bool m_Enable = MGLORIA_VECTORIZATION_FALSE;
};

template<vectorization::VecArch Arch>
struct VecCheck<float, Arch> {
  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<vectorization::VecArch Arch>
struct VecCheck<double, Arch> {
  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, vectorization::VecArch Arch>
struct VecCheck<expr::ScalarExpr<DataType>, Arch> {
  static const bool m_Enable = VecCheck<DataType, Arch>::m_Enable;
};

template<int Dims, typename DataType, vectorization::VecArch Arch>
struct VecCheck<Tensor<CPU, Dims, DataType>, Arch> {
  static const bool m_Enable = VecCheck<DataType, Arch>::m_Enable;
};

template<typename OP, typename A_T, typename DataType, expr::exprType EType,
         vectorization::VecArch Arch>
struct VecCheck<expr::UnaryExpr<OP, A_T, DataType, EType>, Arch> {
  static const bool m_Enable =
      VecCheck<A_T, Arch>::m_Enable && vectorization::VectorizedOP<OP, DataType, Arch>::m_Enable;
};

template<typename OP, typename A_T, typename B_T, typename DataType, expr::exprType EType,
         vectorization::VecArch Arch>
struct VecCheck<expr::BinaryExpr<OP, A_T, B_T, DataType, EType>, Arch> {
  static const bool m_Enable = vectorization::VectorizedOP<OP, DataType, Arch>::m_Enable
                               && VecCheck<A_T, Arch>::m_Enable && VecCheck<B_T, Arch>::m_Enable;
};

// ############################### Below for check Data is ok for Vec ########################
template<int Dims, typename E, vectorization::VecArch Arch>
struct VecDataAlignCheck {
  inline static bool _check(const E& exp) { return false; }
};

template<int Dims, typename DataType, vectorization::VecArch Arch>
struct VecDataAlignCheck<Dims, expr::ScalarExpr<DataType>, Arch> {
  inline static bool _check(const expr::ScalarExpr<DataType>& exp) { return true; }
};

template<int Dims, typename DataType, vectorization::VecArch Arch>
struct VecDataAlignCheck<Dims, Tensor<CPU, Dims, DataType>, Arch> {
  inline static bool _check(const Tensor<CPU, Dims, DataType>& t) {
    return vectorization::NotAlign<Arch>(t.__data_ptr)
           && vectorization::NotAlign<Arch>(t.m_Stride_ * sizeof(DataType));
  }
};

template<int Dims, typename OP, typename A_T, typename DataType, expr::exprType EType,
         vectorization::VecArch Arch>
struct VecDataAlignCheck<Dims, expr::UnaryExpr<OP, A_T, DataType, EType>, Arch> {
  inline static bool _check(const expr::UnaryExpr<OP, A_T, DataType, EType>& t) {
    return VecDataAlignCheck<Dims, A_T, Arch>::_check(t.m_entity);
  }
};

template<int Dims, typename OP, typename A_T, typename B_T, typename DataType, expr::exprType EType,
         vectorization::VecArch Arch>
struct VecDataAlignCheck<Dims, expr::BinaryExpr<OP, A_T, B_T, DataType, EType>, Arch> {
  inline static bool _check(const expr::BinaryExpr<OP, A_T, B_T, DataType, EType>& t) {
    return VecDataAlignCheck<Dims, A_T, Arch>::_check(t.m_lhs)
           && VecDataAlignCheck<Dims, B_T, Arch>::_check(t.m_rhs);
  }
};

}  // namespace mgloria

#endif