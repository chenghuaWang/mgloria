/*!
 *@author   chenghua.wang
 *@file     expression.hpp
 *@brief    This file is heavily depends on expression template.
 *Using curious recurring template pattern(CRTP) to make sure the
 *compute of tensor is efficient and memory friendly.
 */
#include "depends.hpp"
#ifndef _MGLORIA_EXPRESSION_HPP
#define _MGLORIA_EXPRESSION_HPP
namespace mgloria {
/*!
 *@brief Below contain the expression template for tensor to use.
 *If you want to add new OPs. You should follow the CRTP.
 */
namespace expr {
/*!
 *@brief The expression type of each inherited Exp. Each Exp subtype
 * should has one of it.
 */
enum class exprType : uint8_t {
  RValue = 0,   ///! can only be used to assign data.
  Mapped = 1,   ///! element-wise operation.
  Chained = 3,  ///! expression and data that can be chained.
  Complex = 7,  ///! complex operation, such as dot product.
};
// ########################### Basic Template class define. ########################
template<typename SubType, typename DataType, exprType EType>
struct Expression {
  MGLORIA_INLINE_NORMAL SubType* Self() { return static_cast<SubType*>(this); }
  MGLORIA_INLINE_NORMAL const SubType& Self() const { return *static_cast<SubType*>(this); }
};

template<typename LValue, typename RValue, typename DataType>
struct ExpressionDispatcher;
// ########################## Scalar Expression define. #############################
template<typename DataType>
struct ScaleExpr : public Expression<ScaleExpr<DataType>, DataType, exprType::Mapped> {
  ScaleExpr(DataType s) : scale_value(s) {}
  DataType scale_value;
};

template<typename DataType>
MGLORIA_INLINE_NORMAL ScaleExpr<DataType> scale(DataType s) {
  return ScaleExpr<DataType>(s);
}

// ########################## RValue Expression define. #############################
template<typename TrueData, typename DataType>
struct RValue : public Expression<TrueData, DataType, exprType::RValue> {};

}  // namespace expr
}  // namespace mgloria
#endif