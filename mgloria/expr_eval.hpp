#ifndef _MGLORIA_EXPR_EVAL_HPP_
#define _MGLORIA_EXPR_EVAL_HPP_

#pragma once

#include "depends.hpp"
#include "expression.hpp"
#include "prepare.hpp"
#include "tensor.hpp"

namespace mgloria {
namespace expr {

//##############################################################################
//          Below for Job abstract. Which include the expression organization  #
//##############################################################################
/*!
 *@brief
 */
template<typename ExpressionType, typename DataType>
struct Job {
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const;
};
}  // namespace expr
}  // namespace mgloria

#include "complex_eval.hpp"

namespace mgloria {
namespace expr {
// Below is the tensor Job.

/*!
 *@brief
 */
template<typename Device, typename DataType>
struct Job<Tensor<Device, 1, DataType>, DataType> {
  explicit Job(const Tensor<Device, 1, DataType>& T) : __data_ptr(T.__data_ptr) {}

  MGLORIA_INLINE_NORMAL const DataType& Eval(index_t y, index_t x) const { return __data_ptr[x]; }

  MGLORIA_INLINE_NORMAL DataType& REval(index_t y, index_t x) { return __data_ptr[x]; }

 private:
  DataType* __data_ptr;
};

/*!
 *@brief
 */
template<typename Device, int32_t Dims, typename DataType>
struct Job<Tensor<Device, Dims, DataType>, DataType> {
  explicit Job(const Tensor<Device, Dims, DataType>& T)
      : __data_ptr(T.__data_ptr), m_Stride(T.m_Stride_) {}

  MGLORIA_INLINE_NORMAL const DataType& Eval(index_t x, index_t y) const {
    return __data_ptr[y * m_Stride + x];
  }

  MGLORIA_INLINE_NORMAL DataType& REval(index_t x, index_t y) const {
    return __data_ptr[y * m_Stride + x];
  }

 private:
  DataType* __data_ptr;
  index_t m_Stride;
};

// Below is the Transpose Job.
/*!
 *@brief
 */
template<typename E, typename DataType>
struct Job<TransposeExpr<E, DataType>, DataType> {
 public:
  explicit Job(const Job<E, DataType>& _job) : m_job(_job) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const {
    ///! Note that I index the value by (x, y) and Eval is (y, x)
    return m_job.Eval(x, y);
  }

  Job<E, DataType> m_job;
};

// Below is the Scalar Job.
/*!
 *@brief
 */
template<typename DataType>
struct Job<ScalarExpr<DataType>, DataType> {
  explicit Job(DataType s) : m_Scalar(s) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const { return m_Scalar; }

  DataType m_Scalar;
};

// Below is the TypeCast Job.
/*!
 *@brief
 */
template<typename OriDataType, typename DisDataType, typename A_T, exprType EType>
struct Job<TypeCastExpr<OriDataType, DisDataType, A_T, EType>, DisDataType> {
  explicit Job(const Job<A_T, OriDataType>& _job) : m_job(_job) {}
  MGLORIA_INLINE_NORMAL DisDataType Eval(index_t y, index_t x) const {
    return DstDType(m_job.Eval(y, x));
  }

  Job<A_T, OriDataType> m_job;
};

// Below is the Unary Job.
/*!
 *@brief
 */
template<typename OP, typename A_T, typename DataType, exprType EType>
struct Job<UnaryExpr<OP, A_T, DataType, EType>, DataType> {
  explicit Job(const Job<A_T, DataType>& _job) : m_job(_job) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const {
    return OP::Do(m_job.Eval(y, x));
  }

  Job<A_T, DataType> m_job;
};

// Below is the Binary Job.
/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename DataType, exprType EType>
struct Job<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType> {
  explicit Job(const Job<A_T, DataType>& _lhs, const Job<B_T, DataType>& _rhs)
      : m_lhs(_lhs), m_rhs(_rhs) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const {
    return OP::Do(m_lhs.Eval(y, x), m_rhs.Eval(y, x));
  }

  Job<A_T, DataType> m_lhs;
  Job<B_T, DataType> m_rhs;
};

// Below is the Ternary Job.
/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType EType>
struct Job<TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>, DataType> {
  explicit Job(const Job<A_T, DataType>& _1, const Job<B_T, DataType>& _2,
               const Job<C_T, DataType>& _3)
      : m_1(_1), m_2(_2), m_3(_3) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) {
    return OP::Do(m_1.Eval(y, x), m_2.Eval(y, x), m_3.Eval(y, x));
  }

  Job<A_T, DataType> m_1;
  Job<B_T, DataType> m_2;
  Job<C_T, DataType> m_3;
};

// Below is the functions for easily use.
/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename DataType, exprType EType>
MGLORIA_INLINE_NORMAL Job<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType> NewJob(
    const BinaryExpr<OP, A_T, B_T, DataType, EType>& e);

/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType EType>
MGLORIA_INLINE_NORMAL Job<TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>, DataType> NewJob(
    const TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>& e);

/*!
 *@brief
 */
template<typename DataType>
MGLORIA_INLINE_NORMAL Job<ScalarExpr<DataType>, DataType> NewJob(const ScalarExpr<DataType>& e) {
  return Job<ScalarExpr<DataType>, DataType>(e.scale_value);
}

/*!
 *@brief
 */
template<typename OriDataType, typename DisDataType, typename A_T, exprType EType>
MGLORIA_INLINE_NORMAL Job<TypeCastExpr<OriDataType, DisDataType, A_T, EType>, DisDataType> NewJob(
    const TypeCastExpr<OriDataType, DisDataType, A_T, EType>, DisDataType& e) {
  return Job<TypeCastExpr<OriDataType, DisDataType, A_T, EType>, DisDataType>(NewJob(e.m_expr));
}

/*!
 *@brief
 */
template<typename E, typename DataType>
MGLORIA_INLINE_NORMAL Job<E, DataType> NewJob(const RValueExpr<E, DataType>& e) {
  return Job<E, DataType>(e.Self());
}

/*!
 *@brief
 */
template<typename E, typename DataType>
MGLORIA_INLINE_NORMAL Job<TransposeExpr<E, DataType>, DataType> NewJob(
    const TransposeExpr<E, DataType>& e) {
  return Job<TransposeExpr<E, DataType>, DataType>(NewJob(e.m_expr));
}

/*!
 *@brief
 */
template<typename OP, typename A_T, typename DataType, exprType EType>
MGLORIA_INLINE_NORMAL Job<UnaryExpr<OP, A_T, DataType, EType>, DataType> NewJob(
    const UnaryExpr<OP, A_T, DataType, EType>& e) {
  return Job<UnaryExpr<OP, A_T, DataType, EType>, DataType>(NewJob(e.m_entity));
}

/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename DataType, exprType EType>
MGLORIA_INLINE_NORMAL Job<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType> NewJob(
    const BinaryExpr<OP, A_T, B_T, DataType, EType>& e) {
  return Job<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType>(NewJob(e.m_lhs), NewJob(e.m_rhs));
}

/*!
 *@brief
 */
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType EType>
MGLORIA_INLINE_NORMAL Job<TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>, DataType> NewJob(
    const TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>& e) {
  return Job<TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>, DataType>(
      NewJob(e.m_1), NewJob(e.m_2), NewJob(e.m_3));
}

//##############################################################################
//          Below for Basic dispatcher's implementation                        #
//##############################################################################
/*!
 *@brief
 */
template<typename LValue, typename RValue, typename DataType>
struct ExpressionDispatcher {
  template<typename E>
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst,
                                         const Expression<E, DataType, Mapped_t>& exp) {
    MapExpr2Tensor<LValue>(dst, exp);
  }

  template<typename E>
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst,
                                         const Expression<E, DataType, Chained_t>& exp) {
    MapExpr2Tensor<LValue>(dst, exp);
  }

  template<typename E>
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst,
                                         const Expression<E, DataType, RValue_t>& exp) {
    MapExpr2Tensor<LValue>(dst, exp);
  }

  template<typename E>
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst,
                                         const Expression<E, DataType, Complex_t>& exp) {
    ExpressionComplexDispatcher<LValue, RValue, E, DataType>::Eval(dst, exp);
  }
};
}  // namespace expr
}  // namespace mgloria

#endif  // _MGLORIA_EXPR_EVAL_HPP_