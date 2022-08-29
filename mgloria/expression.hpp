/*!
 *@author   chenghua.wang
 *@file     expression.hpp
 *@brief    This file is heavily depends on expression template.
 *Using curious recurring template pattern(CRTP) to make sure the
 *compute of tensor is efficient and memory friendly.
 *@details  All components that will involved in the compute is abstract to expression.
 * For example, A = B + C + D, in this formula C + D will be regarded as a expression whose
 * type is Mapped_t, and B + Expr will be regarded as another expression whose type is also
 * Mapped_t. Finally the A will be regarded as RValue and the result of B + Expr is also the
 * RValue. The RValue will trigger the Eval or to say execute of one expression.
 *
 * All in all, all of data are regarded as expressions here. And all calculate is based on the
 * expression tree generated with the template in compile time.
 */
#ifndef _MGLORIA_EXPRESSION_HPP
#define _MGLORIA_EXPRESSION_HPP
#pragma once
#include "depends.hpp"
#include "prepare_op.hpp"

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
// enum class exprType : uint8_t {
//   RValue = 0,   ///! can only be used to assign data.
//   Mapped = 1,   ///! element-wise operation.
//   Chained = 3,  ///! expression and data that can be chained.
//   Complex = 7,  ///! complex operation, such as dot product.
// };

typedef const int exprType;
exprType RValue_t = 0;   ///! can only be used to assign data.
exprType Mapped_t = 1;   ///! element-wise operation.
exprType Chained_t = 3;  ///! expression and data that can be chained.
exprType Complex_t = 7;  ///! complex operation, such as dot product.

// ########################### Basic Template class define. ########################
/*!
 *@brief      The Basic Template for all Expression.
 *@tparam     SubType record the type inherited the Expression class.
 *@tparam     DataType the datatype in one tensor or expression. float32 by default.
 *@tparam     EType the expression types.
 *@details    All of expression and high-level tensor which need to be computed need
 * inherited from this class. As we all know, A = B + C + D is computed using there
 * base-class object. But the operator +'s behavior is quite different between different
 * expressions, or to say derived class. In that case, I want this base-class object
 * can be cast to the derived class. For that purpose, template can help.
 *
 * The Self() and SelfPtr() member function is quite vital for expression template, they
 * can change the base-class object into derived class object.
 */
template<typename SubType, typename DataType, exprType EType>
struct Expression {
  MGLORIA_INLINE_NORMAL SubType* SelfPtr() { return static_cast<SubType*>(this); }
  MGLORIA_INLINE_NORMAL const SubType& Self() const { return *static_cast<const SubType*>(this); }
};

/*!
 *@brief      The Expression Dispatcher, which respond to actually Eval the expressions.
 *@note       It's only a definition here, for RValueExpr to use. It's implementation is
 * in the expr_eval.hpp file.
 *@tparam     LValue  the LValue should be a class inherited from expression.
 */
template<typename LValue, typename RValue, typename DataType>
struct ExpressionDispatcher;

// ########################## Transpose Expression define.  #########################
/*!
 *@brief      The Transpose expression.
 *@tparam     A_T the expression type passed to this expression. Type should be RValueExpr
 *            TransposeExpr, DotExpr, etc.
 *@tparam     DataType  the element datatype.
 *@note       This class is only an abstract of expression. And this file is only for abstract
 * different expressions. The Eval or to say execute of any operation is not in this file. You
 * can find how Transpose works in expr_eval.hpp. Each operation's implement is different, they
 * will be implemented in different files.
 *
 * This note will be appeared for only once. Other expression in this file is same logic.
 */
template<typename A_T, typename DataType>
struct TransposeExpr : public Expression<TransposeExpr<A_T, DataType>, DataType, Chained_t> {
  explicit TransposeExpr(const A_T& expr) : m_expr(expr) {}

  const A_T& m_expr;

  MGLORIA_INLINE_NORMAL const A_T& T() const { return m_expr; }
};

// ########################## Scalar Expression define. #############################
/*!
 *@brief      The Scalar expression class.
 *@tparam     DataType datatype of one element.
 */
template<typename DataType>
struct ScalarExpr : public Expression<ScalarExpr<DataType>, DataType, Mapped_t> {
  ScalarExpr(DataType s) : scale_value(s) {}
  DataType scale_value;
};

/*!
 *@brief      The Scalar expression function. For easily to use.
 *@tparam     DataType datatype of one element.
 *@param      s The scalar element itself.
 */
template<typename DataType>
MGLORIA_INLINE_NORMAL ScalarExpr<DataType> scalar(DataType s) {
  return ScalarExpr<DataType>(s);
}
// ########################## TypeCast Expression define.   #########################
/*!
 *@brief      Type cast between Original Type and Distribution Type.
 *@tparam     OriDataType original data type.
 *@tparam     DisDataType the type you want to distribute to.
 *@tparam     A_T the expression type passed to this expression. Type should be RValueExpr
 *            TransposeExpr, DotExpr, etc.
 *@tparam     EType the expression classification type. Such as Mapped_t/RValue_t/Chained_t, etc.
 */
template<typename OriDataType, typename DisDataType, typename A_T, exprType EType>
struct TypeCastExpr
    : public Expression<TypeCastExpr<OriDataType, DisDataType, A_T, EType>, DisDataType, EType> {
  explicit TypeCastExpr(const A_T& expr) : m_expr(expr) {}
  const A_T& m_expr;
};

/*!
 *@brief      Type cast between Original Type and Distribution Type class.
 *@tparam     OriDataType original data type.
 *@tparam     DisDataType the type you want to distribute to.
 *@tparam     A_T the expression type passed to this expression. Type should be RValueExpr
 *            TransposeExpr, DotExpr, etc.
 *@tparam     EType the expression classification type. Such as Mapped_t/RValue_t/Chained_t, etc.
 *@param      Expression
 */
template<typename OriDataType, typename DisDataType, typename A_T, exprType EType>
MGLORIA_INLINE_NORMAL TypeCastExpr<OriDataType, DisDataType, A_T, EType | Mapped_t> TypeCast(
    const Expression<A_T, OriDataType, EType>& a) {
  return TypeCastExpr<OriDataType, DisDataType, A_T, EType | Mapped_t>(a.Self());
}

// ########################## RValue Expression define. #############################
/*!
 *@brief
 */
template<typename Container, typename DataType>
struct RValueExpr : public Expression<Container, DataType, RValue_t> {
  // operator overload
  MGLORIA_INLINE_NORMAL Container& operator+=(DataType s) {
    ExpressionDispatcher<op::_plusto, Container, DataType>::Eval(this->SelfPtr(),
                                                                 scalar<DataType>(s));
    return *(this->SelfPtr());
  }

  /*!*/
  MGLORIA_INLINE_NORMAL Container& operator-=(DataType s) {
    ExpressionDispatcher<op::_minusto, Container, DataType>::Eval(this->SelfPtr(),
                                                                  scalar<DataType>(s));
    return *(this->SelfPtr());
  }

  /*!*/
  MGLORIA_INLINE_NORMAL Container& operator*=(DataType s) {
    ExpressionDispatcher<op::_multo, Container, DataType>::Eval(this->SelfPtr(),
                                                                scalar<DataType>(s));
    return *(this->SelfPtr());
  }

  /*!*/
  MGLORIA_INLINE_NORMAL Container& operator/=(DataType s) {
    ExpressionDispatcher<op::_divto, Container, DataType>::Eval(this->SelfPtr(),
                                                                scalar<DataType>(s));
    return *(this->SelfPtr());
  }

  /*!*/
  MGLORIA_INLINE_NORMAL Container& __dispatch(DataType s) {
    ExpressionDispatcher<op::_saveto, Container, DataType>::Eval(this->SelfPtr(),
                                                                 scalar<DataType>(s));
    return *(this->SelfPtr());
  }

  /*!*/
  template<typename A_T, exprType EType>
  MGLORIA_INLINE_NORMAL Container& __dispatch(const Expression<A_T, DataType, EType>& exp) {
    ExpressionDispatcher<op::_saveto, Container, DataType>::Eval(this->SelfPtr(), exp.Self());
    return *(this->SelfPtr());
  }

  /*!*/
  MGLORIA_INLINE_NORMAL Container& __dispatch(const Expression<Container, DataType, RValue_t>& exp);

  // utils functions
  /*!*/
  MGLORIA_INLINE_NORMAL const TransposeExpr<Container, DataType> T() const {
    return TransposeExpr<Container, DataType>(this->self());
  }
};

// ########################## Unary Expression define.  #############################
/*!*/
template<typename OP, typename A_T, typename DataType, exprType EType>
struct UnaryExpr : public Expression<UnaryExpr<OP, A_T, DataType, EType>, DataType, EType> {
  explicit UnaryExpr(const A_T& entity) : m_entity(entity) {}

  const A_T& m_entity;
};

/*!*/
template<typename OP, typename A_T, typename DataType, exprType A_EType>
MGLORIA_INLINE_NORMAL UnaryExpr<OP, A_T, DataType, A_EType | Mapped_t> GenExpr(
    const Expression<A_T, DataType, A_EType>& entity) {
  return UnaryExpr<OP, A_T, DataType, A_EType | Mapped_t>(entity.Self());
}

/*!*/
template<typename OP, typename A_T, typename DataType, exprType A_EType>
MGLORIA_INLINE_NORMAL UnaryExpr<OP, A_T, DataType, A_EType | Mapped_t> Func(
    const Expression<A_T, DataType, A_EType>& entity) {
  return GenExpr<OP>(entity);
}

// ########################## Binary Expression define. #############################
/*!*/
template<typename OP, typename A_T, typename B_T, typename DataType, exprType EType>
struct BinaryExpr : public Expression<BinaryExpr<OP, A_T, B_T, DataType, EType>, DataType, EType> {
  explicit BinaryExpr(const A_T& lhs, const B_T& rhs) : m_lhs(lhs), m_rhs(rhs) {}

  const A_T& m_lhs;
  const B_T& m_rhs;
};

/*!*/
template<typename OP, typename A_T, typename B_T, typename DataType, exprType A_EType,
         exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<OP, A_T, B_T, DataType, A_EType | B_EType | Mapped_t> GenExpr(
    const Expression<A_T, DataType, A_EType>& lhs, const Expression<B_T, DataType, B_EType>& rhs) {
  return BinaryExpr<OP, A_T, B_T, DataType, (A_EType | B_EType | Mapped_t)>(lhs.Self(), rhs.Self());
};

/*!*/
template<typename OP, typename A_T, typename B_T, typename DataType, exprType A_EType,
         exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<OP, A_T, B_T, DataType, A_EType | B_EType | Mapped_t> Func(
    const Expression<A_T, DataType, A_EType>& lhs, const Expression<B_T, DataType, B_EType>& rhs) {
  return GenExpr<OP>(lhs, rhs);
};

/*!*/
template<typename A_T, typename B_T, typename DataType, exprType A_EType, exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<op::_plus, A_T, B_T, DataType, A_EType | B_EType | Mapped_t>
operator+(const Expression<A_T, DataType, A_EType>& lhs,
          const Expression<B_T, DataType, B_EType>& rhs) {
  return Func<op::_plus>(lhs, rhs);
}

/*!*/
template<typename A_T, typename B_T, typename DataType, exprType A_EType, exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<op::_minus, A_T, B_T, DataType, A_EType | B_EType | Mapped_t>
operator-(const Expression<A_T, DataType, A_EType>& lhs,
          const Expression<B_T, DataType, B_EType>& rhs) {
  return Func<op::_minus>(lhs, rhs);
}

/*!*/
template<typename A_T, typename B_T, typename DataType, exprType A_EType, exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<op::_mul, A_T, B_T, DataType, A_EType | B_EType | Mapped_t>
operator*(const Expression<A_T, DataType, A_EType>& lhs,
          const Expression<B_T, DataType, B_EType>& rhs) {
  return Func<op::_mul>(lhs, rhs);
}

/*!*/
template<typename A_T, typename B_T, typename DataType, exprType A_EType, exprType B_EType>
MGLORIA_INLINE_NORMAL BinaryExpr<op::_div, A_T, B_T, DataType, A_EType | B_EType | Mapped_t>
operator/(const Expression<A_T, DataType, A_EType>& lhs,
          const Expression<B_T, DataType, B_EType>& rhs) {
  return Func<op::_div>(lhs, rhs);
}

// ########################## Ternary Expression define. ############################
/*!*/
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType EType>
struct TernaryExpr
    : public Expression<TernaryExpr<OP, A_T, B_T, C_T, DataType, EType>, DataType, EType> {
  explicit TernaryExpr(const A_T& a, const B_T& b, const C_T& c) : m_1(a), m_2(b), m_3(c) {}

  const A_T& m_1;
  const B_T& m_2;
  const C_T& m_3;
};

/*!*/
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType A_EType,
         exprType B_EType, exprType C_EType>
MGLORIA_INLINE_NORMAL
    TernaryExpr<OP, A_T, B_T, C_T, DataType, A_EType | B_EType | C_EType | Mapped_t>
    GenExpr(const Expression<A_T, DataType, A_EType>& _a,
            const Expression<B_T, DataType, B_EType>& _b,
            const Expression<C_T, DataType, C_EType>& _c) {
  return TernaryExpr<OP, A_T, B_T, C_T, DataType, A_EType | B_EType | C_EType | Mapped_t>(
      _a.Self(), _b.Self(), _c.Self());
}

/*!*/
template<typename OP, typename A_T, typename B_T, typename C_T, typename DataType, exprType A_EType,
         exprType B_EType, exprType C_EType>
MGLORIA_INLINE_NORMAL
    TernaryExpr<OP, A_T, B_T, C_T, DataType, A_EType | B_EType | C_EType | Mapped_t>
    Func(const Expression<A_T, DataType, A_EType>& _a, const Expression<B_T, DataType, B_EType>& _b,
         const Expression<C_T, DataType, C_EType>& _c) {
  return GenExpr<OP>(_a, _b, _c);
}
// ########################## Matrix dot Expression define. #########################
/*!*/
template<typename A_T, typename B_T, bool lhs_transposed, bool rhs_transposed, typename DataType>
struct DotExpr : public Expression<DotExpr<A_T, B_T, lhs_transposed, rhs_transposed, DataType>,
                                   DataType, Complex_t> {
  explicit DotExpr(const A_T& a, const B_T& b, DataType scale) : m_a(a), m_b(b), m_scale(scale) {}
  const A_T& m_a;
  const B_T& m_b;
  DataType m_scale;
};

/*!*/
template<typename A_T, typename B_T, bool lhs_transposed, bool rhs_transposed, typename DataType>
MGLORIA_INLINE_NORMAL DotExpr<A_T, B_T, lhs_transposed, rhs_transposed, DataType> batch_dot(
    const RValueExpr<A_T, DataType>& lhs, const RValueExpr<B_T, DataType>& rhs) {
  return DotExpr<A_T, B_T, lhs_transposed, rhs_transposed, DataType>(lhs.Self(), rhs.Self(),
                                                                     DataType(1.f));
}

/*!*/
template<typename A_T, typename B_T, typename DataType>
MGLORIA_INLINE_NORMAL DotExpr<A_T, B_T, false, false, DataType> dot(
    const RValueExpr<A_T, DataType>& lhs, const RValueExpr<B_T, DataType>& rhs) {
  return DotExpr<A_T, B_T, false, false, DataType>(lhs.Self(), rhs.Self(), DataType(1.f));
}

/*!*/
template<typename A_T, typename B_T, typename DataType>
MGLORIA_INLINE_NORMAL DotExpr<A_T, B_T, true, false, DataType> dot(
    const TransposeExpr<A_T, DataType>& lhs, const RValueExpr<B_T, DataType>& rhs) {
  return DotExpr<A_T, B_T, true, false, DataType>(lhs.Self(), rhs.Self(), DataType(1.f));
}

/*!*/
template<typename A_T, typename B_T, typename DataType>
MGLORIA_INLINE_NORMAL DotExpr<A_T, B_T, false, true, DataType> dot(
    const RValueExpr<A_T, DataType>& lhs, const TransposeExpr<B_T, DataType>& rhs) {
  return DotExpr<A_T, B_T, false, true, DataType>(lhs.Self(), rhs.Self(), DataType(1.f));
}

/*!*/
template<typename A_T, typename B_T, typename DataType>
MGLORIA_INLINE_NORMAL DotExpr<A_T, B_T, true, true, DataType> dot(
    const TransposeExpr<A_T, DataType>& lhs, const TransposeExpr<B_T, DataType>& rhs) {
  return DotExpr<A_T, B_T, true, true, DataType>(lhs.Self(), rhs.Self(), DataType(1.f));
}
}  // namespace expr

}  // namespace mgloria

#endif