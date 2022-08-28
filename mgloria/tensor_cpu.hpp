#ifndef _MGLORIA_TENSOR_CPU_HPP_
#define _MGLORIA_TENSOR_CPU_HPP_
#pragma once
#include "tensor.hpp"
#include "./vectorization/__vec_prepare.hpp"

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
  return os;
}

}  // namespace mgloria

#endif