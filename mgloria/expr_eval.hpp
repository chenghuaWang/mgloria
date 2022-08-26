#ifndef _MGLORIA_EXPR_EVAL_HPP_
#define _MGLORIA_EXPR_EVAL_HPP_

#include "depends.hpp"
#include "expression.hpp"
#include "prepare.hpp"
#include "tensor.hpp"

namespace mgloria {
namespace expr {

template<typename ExpressionType, typename DataType>
struct Job {
  MGLORIA_INLINE_NORMAL DataType Eval(index_t x, index_t y) const;
};

template<typename Device, typename DataType>
struct Job<Tensor<Device, 1, DataType>, DataType> {
  explicit Job(const Tensor<Device, 1, DataType>& T) : __data_ptr(T.__data_ptr) {}

  MGLORIA_INLINE_NORMAL const DataType& Eval(index_t x, index_t y) const { return __data_ptr[x]; }

  MGLORIA_INLINE_NORMAL DataType& REval(index_t x, index_t y) const { return __data_ptr[x]; }

 private:
  DataType* __data_ptr;
};

template<typename Device, int32_t Dims, typename DataType>
struct Job<Tensor<Device, Dims, DataType>, DataType> {
  explicit Job(const Tensor<Device, Dims, DataType>& T)
      : __data_ptr(T.__data_ptr), m_Stride(T.m_Stride) {}

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

template<typename LValue, typename RValue, typename DataType>
struct ExpressionDispatcher {
  template<typename E>
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst,
                                         const Expression<E, DataType, Mapped_t>& exp) {
    MapExp<LValue>(dst, exp);
  }
};

}  // namespace expr
}  // namespace mgloria

#endif  // _MGLORIA_EXPR_EVAL_HPP_