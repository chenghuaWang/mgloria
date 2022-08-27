#ifndef _MGLORIA_RUNTIME_CHECK_HPP_
#define _MGLORIA_RUNTIME_CHECK_HPP_
#pragma once
#include "tensor.hpp"
namespace mgloria {
namespace expr {
//##############################################################################
//          Below for Runtime Shape Checking                                   #
//##############################################################################
/*!
 *@brief
 */
template<int32_t Dims, typename E>
struct __runtime_shape_check {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(const E& e);
};

/*!
 *@brief
 */
template<int32_t Dims, typename DataType>
struct __runtime_shape_check<Dims, ScalarExpr<DataType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(const ScalarExpr<DataType>& e) {
    Shape<Dims> __tmp_s__;
#pragma unroll
    for (int i = 0; i < Dims; ++i) { __tmp_s__[i] = 0; }
    return __tmp_s__;
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename OriDataType, typename DisDataType, typename A_T, exprType EType>
struct __runtime_shape_check<Dims, TypeCastExpr<OriDataType, DisDataType, A_T, EType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(
      const TypeCastExpr<OriDataType, DisDataType, A_T, EType>& e) {
    return __runtime_shape_check<Dims, TypeCastExpr<OriDataType, DisDataType, A_T, EType>>::_check(
        e.m_expr);
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename E, typename DataType>
struct __runtime_shape_check<Dims, TransposeExpr<E, DataType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(const E& e) {
    Shape<Dims> __tmp_s__ = __runtime_shape_check<Dims, E>::_check(e.m_expr);
    std::swap(__tmp_s__[0], __tmp_s__[1]);
    return __tmp_s__;
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename DeviceType, typename DataType>
struct __runtime_shape_check<Dims, Tensor<DeviceType, Dims, DataType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(const Tensor<DeviceType, Dims, DataType>& e) {
    return e.m_Shape;
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename OP, typename A_T, typename DataType, exprType EType>
struct __runtime_shape_check<Dims, UnaryExpr<OP, A_T, DataType, EType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(const UnaryExpr<OP, A_T, DataType, EType>& e) {
    return __runtime_shape_check<Dims, UnaryExpr<OP, A_T, DataType, EType>>::_check(e.m_entity);
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename OP, typename A_T, typename B_T, typename DataType, exprType EType>
struct __runtime_shape_check<Dims, BinaryExpr<OP, A_T, B_T, DataType, EType>> {
  MGLORIA_INLINE_NORMAL static Shape<Dims> _check(
      const BinaryExpr<OP, A_T, B_T, DataType, EType>& e) {
    Shape<Dims> __shape_lhs__ = __runtime_shape_check<Dims, A_T>::_check(e.m_lhs);
    Shape<Dims> __shape_rhs__ = __runtime_shape_check<Dims, B_T>::_check(e.m_rhs);
    if (__shape_lhs__[0] == 0) {
      // __shape_lhs__ is scalar.
      return __shape_rhs__;
    }
    if (__shape_rhs__[0] == 0) {
      // __shape_rhs__ is scalar
      return __shape_lhs__;
    }
    CHECK_EQUAL(__shape_lhs__, __shape_rhs__);
    return __shape_lhs__;
  }
};

/*!
 *@brief
 */
template<int32_t Dims, typename OP, typename A_T, typename B_T, typename C_T, typename DataType,
         exprType EType>
struct __runtime_shape_check<Dims, TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>> {
  MGLORIA_INLINE_NORMAL Shape<Dims> _check(
      const TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>& e) {
    Shape<Dims> __shape_1__ = __runtime_shape_check<Dims, A_T>::_check(e.m_1);
    Shape<Dims> __shape_2__ = __runtime_shape_check<Dims, B_T>::_check(e.m_2);
    Shape<Dims> __shape_3__ = __runtime_shape_check<Dims, B_T>::_check(e.m_3);
    if (__shape_1__ == __shape_2__ && __shape_2__ == __shape_3__) {
      LOG_ERR << "__shape_1__ == __shape_2__ && __shape_2__ == __shape_3__"
              << " Found shape1 is" << __shape_1__ << ", shape2 is " << __shape_2__
              << ", shape3 is " << __shape_3__ << std::endl;
      std::exit(MGLORIA_SHAPE_ERROR_EXIT);
    }
    return __shape_1__;
  }
};

//##############################################################################
//          Below for Runtime Device Type Auto Cast and Checking               #
//##############################################################################
// TODO

}  // namespace expr
}  // namespace mgloria
#endif