#ifndef _MGLORIA_PREPARE_OP_H_
#define _MGLORIA_PREPARE_OP_H_
#include "prepare.hpp"

namespace mgloria {
namespace op {
// ################### Binary op ########################################
struct _plus {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return a + b;
  }
};

struct _minus {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return a - b;
  }
};

struct _mul {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return a * b;
  }
};

struct _div {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return a / b;
  }
};

struct _left {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return a;
  }
};

struct _right {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a, DataType b) {
    return b;
  }
};

struct _identity {
  template<typename DataType>
  MGLORIA_INLINE_XPU static DataType Do(DataType a) {
    return a;
  }
};

struct _saveto {
  typedef _right OPType;
  template<typename DataType>
  MGLORIA_INLINE_XPU static void Do(DataType& a, DataType b) {
    a = b;
  }
};

struct _plusto {
  typedef _plus OPType;
  template<typename DataType>
  MGLORIA_INLINE_XPU static void Do(DataType& a, DataType b) {
    a += b;
  }
};

struct _minusto {
  typedef _minus OPType;
  template<typename DataType>
  MGLORIA_INLINE_XPU static void Do(DataType& a, DataType b) {
    a -= b;
  }
};

struct _multo {
  typedef _mul OPType;
  template<typename DataType>
  MGLORIA_INLINE_XPU static void Do(DataType& a, DataType b) {
    a *= b;
  }
};

struct _divto {
  typedef _div OPType;
  template<typename DataType>
  MGLORIA_INLINE_XPU static void Do(DataType& a, DataType b) {
    a /= b;
  }
};
}  // namespace op
}  // namespace mgloria

#endif