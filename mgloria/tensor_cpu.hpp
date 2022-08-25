#ifndef _MGLORIA_TENSOR_CPU_HPP_
#define _MGLORIA_TENSOR_CPU_HPP_
#include "tensor.hpp"

namespace mgloria {

template<typename Device, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL std::ostream& operator<<(std::ostream& os,
                                               const Tensor<Device, Dims, DataType>& T) {
  os << "[Device: CPU]" << T.m_Shape.str();
  return os;
}

}  // namespace mgloria

#endif