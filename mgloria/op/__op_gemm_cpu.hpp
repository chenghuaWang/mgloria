/*!
 *@author chenghua.wang
 *@file   op/__op_gemm_cpu.hpp
 *@brief  The implicit gemm and dot.
 */

#ifndef _MGLORIA___OP_GEMM_CPU_HPP_
#define _MGLORIA___OP_GEMM_CPU_HPP_

///! The #pragma once must be used in this file to prevent recursively include.
#include "prepare.hpp"
#include "tensor_shape.hpp"
#pragma once

#include "../runtime_check.hpp"
#include "../op/__op_cpu.hpp"
#include "../vectorization/__vec_prepare.hpp"

namespace mgloria {
namespace expr {

/*!
 *@brief
 */
template<typename LExpr, typename RExpr, typename DataType>
struct ImplicitGemmExpr;

/*!
 *@brief
 */
template<int Dims, typename LExpr, typename RExpr, typename DataType>
struct __runtime_shape_check<Dims, ImplicitGemmExpr<LExpr, RExpr, DataType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(
      const ImplicitGemmExpr<LExpr, RExpr, DataType>& t) {
    CHECK_EQUAL(Dims, 2);
    Shape<Dims> __tmp_shape_lhs__ = __runtime_shape_check<Dims, LExpr>::_check(t.m_lhs);
    Shape<Dims> __tmp_shape_rhs__ = __runtime_shape_check<Dims, RExpr>::_check(t.m_rhs);
    CHECK_EQUAL(__tmp_shape_lhs__[1], __tmp_shape_rhs__[0]);
    return t.m_out_shape;
  }
};

/*!
 *@brief
 */
template<typename LExpr, typename RExpr, typename DataType>
struct ImplicitGemmExpr
    : public Expression<ImplicitGemmExpr<LExpr, RExpr, DataType>, DataType, Chained_t> {
  ImplicitGemmExpr(const LExpr& lhs, const RExpr& rhs) : m_lhs(lhs), m_rhs(rhs) {
    Shape<2> __tmp_shape_lhs__ = __runtime_shape_check<2, LExpr>::_check(m_lhs);
    Shape<2> __tmp_shape_rhs__ = __runtime_shape_check<2, RExpr>::_check(m_rhs);
    CHECK_EQUAL(__tmp_shape_lhs__[1], __tmp_shape_rhs__[0]);
    m_out_shape = makeShape2d(__tmp_shape_lhs__[0], __tmp_shape_rhs__[1]);
    m_internal_size = __tmp_shape_lhs__[1];
  }

  const LExpr& m_lhs;
  const RExpr& m_rhs;
  Shape<2> m_out_shape;
  index_t m_internal_size;
};

/*!
 *@brief
 */
template<typename LExpr, typename RExpr, typename DataType, exprType e1, exprType e2>
MGLORIA_INLINE_NORMAL ImplicitGemmExpr<LExpr, RExpr, DataType> implicit_dot(
    const Expression<LExpr, DataType, e1>& lhs, const Expression<RExpr, DataType, e2>& rhs) {
  return ImplicitGemmExpr<LExpr, RExpr, DataType>(lhs.Self(), rhs.Self());
}

/*!
 *@brief
 */
template<typename LExpr, typename RExpr, typename DataType>
struct Job<ImplicitGemmExpr<LExpr, RExpr, DataType>, DataType> {
  explicit Job(const ImplicitGemmExpr<LExpr, RExpr, DataType>& e)
      : m_lhs(e.m_lhs),
        m_rhs(e.m_rhs),
        m_internal_size(e.m_internal_size),
        m_internal_size_floor_aligned(
            vectorization::FloorAlign<MGLORIA_VECTORIZATION_ARCH, DataType>(e.m_internal_size)) {}
  MGLORIA_INLINE_NORMAL DataType Eval(index_t y, index_t x) const {
    using namespace vectorization;
    Vectorized<DataType> __vec__ = Vectorized<DataType>::Fill(0);

    const size_t __vec_size__ = Vectorized<DataType>::num;
    DataType __lhs__[__vec_size__], __rhs__[__vec_size__];

    for (index_t i = 0; i < m_internal_size_floor_aligned; i += __vec_size__) {
      for (index_t j = 0; j < __vec_size__; ++j) { __lhs__[j] = m_lhs.Eval(y, i + j); }
      for (index_t j = 0; j < __vec_size__; ++j) { __rhs__[j] = m_rhs.Eval(i + j, x); }
      __vec__ = __vec__
                + Vectorized<DataType>::LoadUnAligned(__lhs__)
                      * Vectorized<DataType>::LoadUnAligned(__rhs__);
    }

    DataType res = __vec__.Sum();

    for (index_t i = m_internal_size_floor_aligned; i < m_internal_size; ++i) {
      res = res + m_lhs.Eval(y, i) * m_rhs.Eval(i, x);
    }
    return res;
  }

 private:
  Job<LExpr, DataType> m_lhs;
  Job<RExpr, DataType> m_rhs;
  const index_t m_internal_size;
  const index_t m_internal_size_floor_aligned;
};

template<typename LExpr, typename RExpr, typename DataType>
inline Job<ImplicitGemmExpr<LExpr, RExpr, DataType>, DataType> NewJob(
    const ImplicitGemmExpr<LExpr, RExpr, DataType>& e) {
  return Job<ImplicitGemmExpr<LExpr, RExpr, DataType>, DataType>(e);
}

}  // namespace expr
}  // namespace mgloria

#endif  // _MGLORIA___OP_GEMM_CPU_HPP_