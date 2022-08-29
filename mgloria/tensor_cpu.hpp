#ifndef _MGLORIA_TENSOR_CPU_HPP_
#define _MGLORIA_TENSOR_CPU_HPP_
#pragma once

#include <iomanip>

#include "tensor.hpp"
#include "./vectorization/__vec_prepare.hpp"

#include "expr_eval.hpp"

namespace mgloria {

template<>
MGLORIA_INLINE_NORMAL void InitTensorComputeMachine<CPU>(int devIdx) {
  LOG_WARN << "CPU type based no need to init" << std::endl;
}

template<>
MGLORIA_INLINE_NORMAL void ShutdownTensorComputeMachine<CPU>(int devIdx) {
  LOG_WARN << "CPU type based no need to shutdown" << std::endl;
}

template<>
MGLORIA_INLINE_NORMAL void SetCurrentDevice<CPU>(int devIdx) {
  LOG_WARN << "Actually, we have not support multi physical CPU\n";
}

template<>
MGLORIA_INLINE_NORMAL void FreeStream(Stream<CPU>* stream) {
  delete stream;
}

template<>
MGLORIA_INLINE_NORMAL Stream<CPU>* NewStream(index_t devIdx) {
  return new Stream<CPU>;
}

template<typename Device, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL std::ostream& operator<<(std::ostream& os,
                                               const Tensor<Device, Dims, DataType>& T) {
  os << "[Device: CPU]" << T.m_Shape.str();
  if (T.__data_ptr == nullptr) { os << "The data ptr in Tensor is nullptr."; }
  os << "[\n";
  auto __tmp_shape__ = T.m_Shape.Flatten2D();
  bool reduced_r = false;
  bool reduced_c = false;
  index_t end_r = __tmp_shape__[0];
  index_t end_c = __tmp_shape__[1];
  if (__tmp_shape__[0] > MGLORIA_MAX_SHOW_LENGTH) {
    reduced_r = true;
    end_r = MGLORIA_MAX_SHOW_LENGTH;
  }
  if (__tmp_shape__[1] > MGLORIA_MAX_SHOW_LENGTH) {
    reduced_c = true;
    end_c = MGLORIA_MAX_SHOW_LENGTH;
  }
#pragma unroll
  for (index_t _r = 0; _r < end_r; ++_r) {
    for (index_t _c = 0; _c < end_c; ++_c) {
      os << std::setiosflags(std::ios::scientific | std::ios::showpos | std::ios::right)
         << T.__data_ptr[_r * __tmp_shape__[0] + _c] << ", ";
    }
    os << '\n';
  }
  os << "]\n";
  os << std::resetiosflags(std::ios::scientific | std::ios::showpos | std::ios::right);
  return os;
}

}  // namespace mgloria

#endif