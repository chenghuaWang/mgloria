/*!
 *@author   chenghua.wang
 *@file     tensor_stream.hpp
 *@brief    Define the Device Type. The stream for asynchronous execute.
 *@note     This file only has defs, implementation is in stream_gpu.hpp
 */

#ifndef _MGLORIA_TENSOR_STREAM_HPP_
#define _MGLORIA_TENSOR_STREAM_HPP_
#pragma once
#include "depends.hpp"

namespace mgloria {

enum class DeviceType : uint8_t {
  ERR_T = 0,
  CPU_T = 1,
  GPU_T = 2,
  AUTO_T = 3,
};

struct CPU {
  static const bool usingCPU = true;
  static const bool usingGPU = false;
  static const bool AutoDevice = false;
  static const DeviceType devType = DeviceType::CPU_T;
};

struct GPU {
  static const bool usingCPU = false;
  static const bool usingGPU = true;
  static const bool AutoDevice = false;
  static const DeviceType devType = DeviceType::GPU_T;
};

struct XPU {
  static const bool usingCPU = true;
  static const bool usingGPU = true;
  static const bool AutoDevice = true;
  static const DeviceType devType = DeviceType::AUTO_T;
};

const DeviceType default_device_t = DeviceType::CPU_T;

/*!
 *@brief    A naive implementation for CPU. GPU's implementation
 * is in stream_gpu.hpp.
 */
template<typename device>
struct Stream {
  MGLORIA_INLINE_NORMAL void Wait() {}
  MGLORIA_INLINE_NORMAL bool IsIdle() { return true; }
  MGLORIA_INLINE_NORMAL void CreateBlasHandle() {}
};

template<typename device>
MGLORIA_INLINE_NORMAL void FreeStream(Stream<device>* stream);

template<typename device>
MGLORIA_INLINE_NORMAL Stream<device>* NewStream(index_t devIdx);

}  // namespace mgloria

#endif  // _MGLORIA_TENSOR_STREAM_HPP_