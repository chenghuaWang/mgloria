#ifndef _MGLORIA_TENSOR_CPU_HPP_
#define _MGLORIA_TENSOR_CPU_HPP_

#pragma once

#include <iomanip>

#include "tensor.hpp"
#include "./vectorization/veced_op.hpp"

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

template<typename DeviceType, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL std::ostream& operator<<(std::ostream& os,
                                               const Tensor<DeviceType, Dims, DataType>& T) {
  os << "[DeviceType: CPU]" << T.m_Shape.str();
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

// ############################### Below for memory allocate. #####################

template<typename xpu>
MGLORIA_INLINE_NORMAL void* __HostMalloc__(size_t size);

template<typename xpu>
MGLORIA_INLINE_NORMAL void __HostFree__(void* ptr);

#if MGLORIA_USE_CUDA == 1 && defined __CUDACC__

template<>
MGLORIA_INLINE_NORMAL void* __HostMalloc__<GPU>(size_t size) {
  // CUDA call function, but memory is allocated in host.
  void* ptr;
  GUARD_CUDA_CALL(cudaMallocHost(&ptr, size, cudaHostAllocPortable));
  return ptr;
}

template<>
MGLORIA_INLINE_NORMAL void __HostFree__<GPU>(void* ptr) {
  // CUDA call function, but memory is freed in host.
  GUARD_CUDA_CALL(cudaFreeHost(ptr));
}
#endif  // MGLORIA_USE_CUDA == 1 && defined __CUDACC__

template<>
MGLORIA_INLINE_NORMAL void* __HostMalloc__<CPU>(size_t size) {
  size_t pitch;
  return vectorization::MallocAlignedPitch(&pitch, size, 1);
}

template<>
MGLORIA_INLINE_NORMAL void __HostFree__<CPU>(void* ptr) {
  vectorization::FreeAlignedPitch(ptr);
}

template<typename xpu, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void HostMalloc(Tensor<CPU, Dims, DataType>* T) {
  T->m_Stride_ = T->size(Dims - 1);
  CHECK_EQUAL(T->IsContiguous(), true);
  void* ptr = __HostMalloc__<xpu>(T->AllElementNum() * sizeof(DataType));
  T->__data_ptr = reinterpret_cast<DataType*>(ptr);
}

template<typename xpu, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void HostFree(Tensor<CPU, Dims, DataType>* T) {
  if (T->__data_ptr == nullptr) { LOG_WARN << "Double Free.\n"; }
  __HostFree__<xpu>(T->__data_ptr);
  T->__data_ptr = nullptr;
}

// ######################## Below for Tensor memory allocate. #####################

template<int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void HostMallocTensorMem(Tensor<CPU, Dims, DataType>* T, bool pad) {
  size_t pitch;
  void* ptr;
  if (pad) {
    ptr = vectorization::MallocAlignedPitch(&pitch, T->size(Dims - 1) * sizeof(DataType),
                                            T->m_Shape.Flatten2D()[0]);
    T->m_Stride_ = static_cast<index_t>(pitch / sizeof(DataType));
  } else {
    T->m_Stride_ = T->size(Dims - 1);
    ptr = vectorization::MallocAlignedPitch(&pitch, T->m_Shape.Size() * sizeof(DataType), 1);
  }
  T->__data_ptr = reinterpret_cast<DataType*>(ptr);
}

template<int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void HostFreeTensorMem(Tensor<CPU, Dims, DataType>* T) {
  vectorization::FreeAlignedPitch(T->__data_ptr);
  T->__data_ptr = nullptr;
}

template<typename DeviceType, typename DataType, int Dims>
MGLORIA_INLINE_NORMAL Tensor<DeviceType, Dims, DataType> NewTensor(const Shape<Dims>& shape,
                                                                   bool init, DataType InitValue,
                                                                   bool pad,
                                                                   Stream<DeviceType>* stream_) {
  Tensor<DeviceType, Dims, DataType> T(shape);
  T.m_Stream = stream_;
  HostMallocTensorMem(&T, pad);
  if (init) { T = InitValue; }
  return T;
}

template<typename DeviceType, typename DataType, int Dims>
MGLORIA_INLINE_NORMAL void DeleteTensor(Tensor<DeviceType, Dims, DataType>* T) {
  HostFreeTensorMem(T);
}

// ######################## Below for actual tensor expression execute ############
template<typename Saver, typename R, int Dims, typename DataType, typename E>
MGLORIA_INLINE_NORMAL void MapJob2Tensor(TRValue<R, CPU, Dims, DataType>* dst,
                                         const expr::Job<E, DataType>& plan) {
  Shape<2> __shape__ = expr::__runtime_shape_check<Dims, R>::_check(dst->Self()).Flatten2D();
  expr::Job<R, DataType> disJobs = expr::NewJob(dst->Self());
#pragma omp parallel for
  for (openmp_index_t y = 0; y < __shape__[0]; ++y) {
    for (index_t x = 0; x < __shape__[1]; ++x) {
      Saver::template Do<DataType>(disJobs.REval(y, x), plan.Eval(y, x));
    }
  }
}

template<bool Passed, typename Saver, typename R, int Dims, typename DataType, typename E,
         int etype>
struct MapExpr2Tensor_CPU {
  MGLORIA_INLINE_NORMAL static void Do(TRValue<R, CPU, Dims, DataType>* dst,
                                       const expr::Expression<E, DataType, etype>& exp) {
    MapJob2Tensor<Saver>(dst, expr::NewJob(exp.Self()));
  }
};

template<typename SV, int Dims, typename DataType, typename E, int etype>
struct MapExpr2Tensor_CPU<true, SV, Tensor<CPU, Dims, DataType>, Dims, DataType, E, etype> {
  MGLORIA_INLINE_NORMAL static void Do(Tensor<CPU, Dims, DataType>* dst,
                                       const expr::Expression<E, DataType, etype>& exp) {
    if (VecDataAlignCheck<Dims, E, MGLORIA_VECTORIZATION_ARCH>::_check(exp.Self())
        && VecDataAlignCheck<Dims, Tensor<CPU, Dims, DataType>, MGLORIA_VECTORIZATION_ARCH>::_check(
            *dst)) {
      expr::ExecuteVectorizedJob<SV>(
          dst->Self(), expr::NewVectorizedJob<MGLORIA_VECTORIZATION_ARCH>(exp.Self()));
    } else {
      MapJob2Tensor<SV>(dst, expr::NewJob(exp.Self()));
    }
  }
};

template<typename Saver, typename R, int Dims, typename DType, typename E, int etype>
MGLORIA_INLINE_NORMAL void MapExpr2Tensor(TRValue<R, CPU, Dims, DType>* dst,
                                          const expr::Expression<E, DType, etype>& exp) {
  MapExpr2Tensor_CPU<VecCheck<E, MGLORIA_VECTORIZATION_ARCH>::m_Enable, Saver, R, Dims, DType, E,
                     etype>::Do(dst->SelfPtr(), exp);
}

}  // namespace mgloria

#endif