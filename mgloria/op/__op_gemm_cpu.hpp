#ifndef _MGLORIA___OP_GEMM_CPU_HPP_
#define _MGLORIA___OP_GEMM_CPU_HPP_
#pragma once
#include "../runtime_check.hpp"
#include "./op/__op_cpu.hpp"
#include "../vectorization/__vec_prepare.hpp"

namespace mgloria {
namespace expr {

template<typename LExpr, typename RExpr, typename DataType>
struct GemmExpr : public Expression<GemmExpr<LExpr, RExpr, DataType>, DataType, Chained_t> {
  GemmExpr(const LExpr& lhs, const RExpr& rhs) : m_lhs(lhs), m_rhs(rhs) {
    Shape<2> __tmp_shape_lhs__ = __runtime_shape_check<2, LExpr>::_check(m_lhs);
    Shape<2> __tmp_shape_rhs__ = __runtime_shape_check<2, RExpr>::_check(m_rhs);
  }

  const LExpr& m_lhs;
  const RExpr& m_rhs;
  Shape<2> m_out_shape;
};

}  // namespace expr
}  // namespace mgloria

#endif  // _MGLORIA___OP_GEMM_CPU_HPP_