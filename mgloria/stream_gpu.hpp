/*!
 *@author   chenghua.wang
 *@file     stream_gpu.hpp
 *@brief    The Stream implementation for GPU. Definition
 * can be found in tensor_stream.hpp
 */

#ifndef _MGLORIA_STREAM_GPU_HPP_
#define _MGLORIA_STREAM_GPU_HPP_
#pragma once
#include "tensor_stream.hpp"
#if MGLORIA_USE_CUDA == 1
#include <memory>
namespace mgloria {

enum class HandleStatus : uint8_t {
  no = 0,
  own = 1,
};

#if MGLORIA_USE_CUDNN == 1
template<typename device>
MGLORIA_INLINE_NORMAL Stream<device>* NewStream(bool create_dnn = false, index_t devIdx);
#endif  // MGLORIA_USE_CUDNN == 1

/*!
 *@brief        The Stream implementation in CUDA. The original definition is in
 * tensor_stream.hpp file.
 */
template<>
struct Stream<GPU> {
  // ########################## Constructor ###############################
  Stream()
      : m_Stream(0),
        m_BlasHandle(0),
#if MGLORIA_USE_CUDNN == 1
#endif  // MGLORIA_USE_CUDNN == 1
        m_Stream_ownership(HandleStatus::no),
        m_BlasHandle_ownership(HandleStatus::no) {
  }
  // ########################## parameter contain #########################
 public:
  cudaDeviceProp m_properties;
  cudaStream_t m_Stream;
  cublasHandle_t m_BlasHandle;
#if MGLORIA_USE_CUDNN == 1
#endif  // MGLORIA_USE_CUDNN == 1
  HandleStatus m_Stream_ownership;
  HandleStatus m_BlasHandle_ownership;

 public:
  index_t deviceID = 0;
  // ########################## Utils functions ###########################
  MGLORIA_INLINE_NORMAL void Wait() { GUARD_CUDA_CALL(cudaStreamSynchronize(m_Stream)); }

  MGLORIA_INLINE_NORMAL bool IsIdle() {
    cudaError_t err = cudaStreamQuery(m_Stream);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    LOG_ERR << cudaGetErrorString(err);
    return false;
  }

  MGLORIA_INLINE_NORMAL void FreeBlasHandle() {
    if (m_BlasHandle_ownership == HandleStatus::own) {
      cublasStatus_t err = cublasDestroy(m_BlasHandle);
      m_BlasHandle_ownership = HandleStatus::no;
      CHECK_EQUAL(err, CUBLAS_STATUS_SUCCESS);
    }
  }

  MGLORIA_INLINE_NORMAL void CreateBlasHandle() {
    FreeBlasHandle();
    cublasStatus_t err = cublasCreate(&m_BlasHandle);
    m_BlasHandle_ownership = HandleStatus::own;
    CHECK_EQUAL(err, CUBLAS_STATUS_SUCCESS);
    err = cublasSetStream(m_BlasHandle, m_Stream);
    CHECK_EQUAL(err, CUBLAS_STATUS_SUCCESS);
  }

  MGLORIA_INLINE_NORMAL static cublasHandle_t& GetBlasHandle(Stream<GPU>* s) {
    return s->m_BlasHandle;
  };

  MGLORIA_INLINE_NORMAL static cudaStream_t& GetCUDAStream(Stream<GPU>* s) { return s->m_Stream; };
};

/*!
 *@brief        Delete CUDA Stream. Not support for cuDNN.
 *@param        s Stream<GPU>'s pointer.
 */
template<>
MGLORIA_INLINE_NORMAL void FreeStream<GPU>(Stream<GPU>* s) {
  GUARD_CUDA_CALL(cudaStreamDestroy(s->m_Stream));
  s->FreeBlasHandle();
#if MGLORIA_USE_CUDNN == 1
// TODO
#endif  // MGLORIA_USE_CUDNN == 1
  delete s;
}

/*!
 *@brief        Create CUDA Stream. No cuDNN support.
 *@param        devIdx the device's idx
 *@return       Stream<GPU>'s pointer.
 *@note         In this implementation. RAII is used to avoid CUDA exception lead memory leak.
 */
template<>
MGLORIA_INLINE_NORMAL Stream<GPU>* NewStream(index_t devIdx) {
  struct StreamDestroyer {
    void operator()(Stream<GPU>* s) const { FreeStream<GPU>(s); }
  };
  std::unique_ptr<Stream<GPU>, StreamDestroyer> st(new Stream<GPU>());
  GUARD_CUDA_CALL(cudaStreamCreate(&st->m_Stream));
  st->CreateBlasHandle();
  st->deviceID = devIdx;
  if (devIdx != -1) { GUARD_CUDA_CALL(cudaGetDeviceProperties(&st->m_properties, devIdx)); }
  return st.release();
}

#if MGLORIA_USE_CUDNN == 1
/*!
 *@brief        Create CUDA Stream. With cuDNN support.
 *@param        devIdx the device's idx
 *@return       Stream<GPU>'s pointer.
 *@note         In this implementation. RAII is used to avoid CUDA exception lead memory leak.
 */
template<>
MGLORIA_INLINE_NORMAL Stream<GPU>* NewStream(bool create_dnn = false, index_t devIdx) {
  // TODO
}
#endif  // MGLORIA_USE_CUDNN == 1

}  // namespace mgloria
#endif  // MGLORIA_USE_CUDA == 1
#endif  // _MGLORIA_STREAM_GPU_HPP_