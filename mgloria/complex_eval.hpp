/*!
 *@author chenghua.wang
 *@file   complex_eval.hpp
 *@brief  The ComplexExpr dispatcher and dotExpr dispatcher.
 * Highly reference from MShadow. Need to be changed. For now
 * is to eval the code corrections.
 */

#ifndef _MGLORIA_COMPLEX_EVAL_HPP_
#define _MGLORIA_COMPLEX_EVAL_HPP_
#pragma once

#include "depends.hpp"

#if (MGLORIA_USE_MKL == 1 && INTEL_MKL_VERSION >= 20160000)
#include <vector>
#endif

#include "op/__op_gemm_cpu.hpp"
#if MGLORIA_USE_CUDA == 1
#ifdef __CUDACC__
#include "cuda/tensor_gpu.cuh"
#endif
#endif

namespace mgloria {

/*!
 *@brief
 */
template<typename DeviceType, typename DataType>
MGLORIA_INLINE_NORMAL void GetBatchedView(DataType** dst, DataType* src, int num, int stride,
                                          Stream<DeviceType>* stream);

/*!
 *@brief
 */
template<typename DataType>
MGLORIA_INLINE_NORMAL void GetBatchedView(DataType** dst, DataType* src, int num, int stride,
                                          Stream<CPU>* stream) {
  for (int i = 0; i < num; i++) { dst[i] = src + i * stride; }
}
// #ifdef __CUDACC__
// namespace cuda {};
// template<typename DataType>
// MGLORIA_INLINE_NORMAL void GetBatchedView(DataType** dst, DataType* src, int num, int stride,
//                                           Stream<GPU>* stream) {
//   cuda::GetBatchedView(dst, src, num, stride, stream);
// }
// #endif
// TODO

namespace expr {
/*!
 *@brief
 */
template<typename LValue, typename RValue, typename E, typename DataType>
struct ExpressionComplexDispatcher {
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst, const E& exp);
};

/*!
 *@brief
 */
template<typename LValue, typename DeviceType, int DisDims, int LeftDims, int RightDims,
         bool LeftTransposed, bool RightTransposed, typename DataType>
struct DotEngine {
  MGLORIA_INLINE_NORMAL static void Eval(Tensor<DeviceType, DisDims, DataType>* p_dst,
                                         const Tensor<DeviceType, LeftDims, DataType>& lhs,
                                         const Tensor<DeviceType, RightDims, DataType>& rhs,
                                         DataType scale);
};

// handles the dot, use CblasColMajor
template<typename DeviceType, typename DType = float>
struct BLASEngine {
  MGLORIA_INLINE_NORMAL static bool GetT(bool t) { return t ? true : false; }
  MGLORIA_INLINE_NORMAL static void SetStream(Stream<DeviceType>* stream) {}
  MGLORIA_INLINE_NORMAL static void gemm(Stream<DeviceType>* stream, bool transa, bool transb,
                                         int m, int n, int k, DType alpha, const DType* A, int lda,
                                         const DType* B, int ldb, DType beta, DType* C, int ldc) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void batched_gemm(Stream<DeviceType>* stream, bool transa,
                                                 bool transb, int m, int n, int k, DType alpha,
                                                 const DType* A, int lda, const DType* B, int ldb,
                                                 DType beta, DType* C, int ldc, int batch_count,
                                                 DType** workspace) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void gemv(Stream<DeviceType>* stream, bool trans, int m, int n,
                                         DType alpha, const DType* A, int lda, const DType* X,
                                         int incX, DType beta, DType* Y, int incY) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void batched_gemv(Stream<DeviceType>* stream, bool trans, int m,
                                                 int n, DType alpha, const DType* A, int lda,
                                                 const DType* X, int incX, DType beta, DType* Y,
                                                 int incY, int batch_count) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void ger(Stream<DeviceType>* stream, int m, int n, DType alpha,
                                        const DType* X, int incX, const DType* Y, int incY,
                                        DType* A, int lda) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void batched_ger(Stream<DeviceType>* stream, int m, int n,
                                                DType alpha, const DType* X, int incX,
                                                const DType* Y, int incY, DType* A, int lda,
                                                int batch_count) {
    LOG_ERR << "Not implemented!";
  }
  MGLORIA_INLINE_NORMAL static void dot(Stream<DeviceType>* stream, int n, const DType* X, int incX,
                                        const DType* Y, int incY, DType* ret) {
    LOG_ERR << "Not implemented!";
  }
};

#if MGLORIA_USE_MKL == 1
template<>
struct BLASEngine<CPU, float> {
  inline static CBLAS_TRANSPOSE GetT(bool t) { return t ? CblasTrans : CblasNoTrans; }
  inline static void SetStream(Stream<CPU>* stream) {}
  inline static void gemm(Stream<CPU>* stream, bool transa, bool transb, int m, int n, int k,
                          float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                          float* C, int ldc) {
    cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb), m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
  }
  inline static void batched_gemm(Stream<CPU>* stream, bool transa, bool transb, int m, int n,
                                  int k, float alpha, const float* A, int lda, const float* B,
                                  int ldb, float beta, float* C, int ldc, int batch_count,
                                  float** workspace) {
#if (MGLORIA_USE_MKL && INTEL_MKL_VERSION >= 20160000)
    // since same m/n/k is used for all single gemms, so we put all gemms into one group
    const int GROUP_SIZE = 1;
    MKL_INT p_m[GROUP_SIZE] = {m};
    MKL_INT p_n[GROUP_SIZE] = {n};
    MKL_INT p_k[GROUP_SIZE] = {k};
    MKL_INT p_lda[GROUP_SIZE] = {lda};
    MKL_INT p_ldb[GROUP_SIZE] = {ldb};
    MKL_INT p_ldc[GROUP_SIZE] = {ldc};

    float p_alpha[GROUP_SIZE] = {alpha};
    float p_beta[GROUP_SIZE] = {beta};

    CBLAS_TRANSPOSE cblas_a_trans = GetT(transa);
    CBLAS_TRANSPOSE cblas_b_trans = GetT(transb);

    MKL_INT p_group_sizeb[GROUP_SIZE] = {batch_count};
    CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
    CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

    std::vector<const float*> pp_A;
    std::vector<const float*> pp_B;
    std::vector<float*> pp_C;
    pp_A.reserve(batch_count);
    pp_B.reserve(batch_count);
    pp_C.reserve(batch_count);

    auto m_k = m * k;
    auto k_n = k * n;
    auto m_n = m * n;

    for (int i = 0; i < batch_count; i++) {
      pp_A[i] = A + i * m_k;
      pp_B[i] = B + i * k_n;
      pp_C[i] = C + i * m_n;
    }

    cblas_sgemm_batch(CblasColMajor, p_transa, p_transb, p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda,
                      pp_B.data(), p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha, A + i * m * k, lda, B + i * k * n, ldb, beta,
           C + i * m * n, ldc);
    }
#endif
  }
  inline static void gemv(Stream<CPU>* stream, bool trans, int m, int n, float alpha,
                          const float* A, int lda, const float* X, int incX, float beta, float* Y,
                          int incY) {
    cblas_sgemv(CblasColMajor, GetT(trans), m, n, alpha, A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<CPU>* stream, bool trans, int m, int n, float alpha,
                                  const float* A, int lda, const float* X, int incX, float beta,
                                  float* Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda, X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<CPU>* stream, int m, int n, float alpha, const float* X, int incX,
                         const float* Y, int incY, float* A, int lda) {
    cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<CPU>* stream, int m, int n, float alpha, const float* X,
                                 int incX, const float* Y, int incY, float* A, int lda,
                                 int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY, A + i * lda * n,
          lda);
    }
  }
  inline static void dot(Stream<CPU>* stream, int n, const float* X, int incX, const float* Y,
                         int incY, float* ret) {
    *ret = cblas_sdot(n, X, incX, Y, incY);
  }
};

template<>
struct BLASEngine<CPU, double> {
  inline static CBLAS_TRANSPOSE GetT(bool t) { return t ? CblasTrans : CblasNoTrans; }
  inline static void SetStream(Stream<CPU>* stream) {}
  inline static void gemm(Stream<CPU>* stream, bool transa, bool transb, int m, int n, int k,
                          double alpha, const double* A, int lda, const double* B, int ldb,
                          double beta, double* C, int ldc) {
    cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb), m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
  }
  inline static void batched_gemm(Stream<CPU>* stream, bool transa, bool transb, int m, int n,
                                  int k, double alpha, const double* A, int lda, const double* B,
                                  int ldb, double beta, double* C, int ldc, int batch_count,
                                  double** workspace) {
#if (MGLORIA_USE_MKL == 1 && INTEL_MKL_VERSION >= 20160000)
    // since same m/n/k is used for all single gemms, so we put all gemms into one group
    const int GROUP_SIZE = 1;
    MKL_INT p_m[GROUP_SIZE] = {m};
    MKL_INT p_n[GROUP_SIZE] = {n};
    MKL_INT p_k[GROUP_SIZE] = {k};
    MKL_INT p_lda[GROUP_SIZE] = {lda};
    MKL_INT p_ldb[GROUP_SIZE] = {ldb};
    MKL_INT p_ldc[GROUP_SIZE] = {ldc};

    double p_alpha[GROUP_SIZE] = {alpha};
    double p_beta[GROUP_SIZE] = {beta};

    CBLAS_TRANSPOSE cblas_a_trans = GetT(transa);
    CBLAS_TRANSPOSE cblas_b_trans = GetT(transb);

    MKL_INT p_group_sizeb[GROUP_SIZE] = {batch_count};
    CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
    CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

    std::vector<const double*> pp_A;
    std::vector<const double*> pp_B;
    std::vector<double*> pp_C;
    pp_A.reserve(batch_count);
    pp_B.reserve(batch_count);
    pp_C.reserve(batch_count);

    auto m_k = m * k;
    auto k_n = k * n;
    auto m_n = m * n;

    for (int i = 0; i < batch_count; i++) {
      pp_A[i] = A + i * m_k;
      pp_B[i] = B + i * k_n;
      pp_C[i] = C + i * m_n;
    }

    cblas_dgemm_batch(CblasColMajor, p_transa, p_transb, p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda,
                      pp_B.data(), p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha, A + i * m * k, lda, B + i * k * n, ldb, beta,
           C + i * m * n, ldc);
    }
#endif
  }
  inline static void gemv(Stream<CPU>* stream, bool trans, int m, int n, double alpha,
                          const double* A, int lda, const double* X, int incX, double beta,
                          double* Y, int incY) {
    cblas_dgemv(CblasColMajor, GetT(trans), m, n, alpha, A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<CPU>* stream, bool trans, int m, int n, double alpha,
                                  const double* A, int lda, const double* X, int incX, double beta,
                                  double* Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda, X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<CPU>* stream, int m, int n, double alpha, const double* X, int incX,
                         const double* Y, int incY, double* A, int lda) {
    cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<CPU>* stream, int m, int n, double alpha, const double* X,
                                 int incX, const double* Y, int incY, double* A, int lda,
                                 int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY, A + i * lda * n,
          lda);
    }
  }
  inline static void dot(Stream<CPU>* stream, int n, const double* X, int incX, const double* Y,
                         int incY, double* ret) {
    *ret = cblas_ddot(n, X, incX, Y, incY);
  }
};
#elif MGLORIA_USE_CUDA == 1
#ifdef __CUDACC__
// TODO
#endif  //__CUDACC__
#else
// TODO
#endif  // MGLORIA_USE_MKL == 1

inline Shape<2> GetShape(const Shape<2>& shape, bool transpose) {
  return transpose ? makeShape2d(shape[1], shape[0]) : shape;
}

template<typename SV, typename xpu, bool transpose_left, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 2, 2, 2, transpose_left, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 2, DType>* p_dst, const Tensor<xpu, 2, DType>& lhs,
                          const Tensor<xpu, 2, DType>& rhs, DType scale) {
    Tensor<xpu, 2, DType>& dst = *p_dst;
#if MGLORIA_USE_CUDA != 1 && MGLORIA_USE_MKL != 1
    if (xpu::kDevMask == cpu::kDevMask && scale == 1.0f) {
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs);
        return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T());
        return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs);
        return;
      }
    }
#endif
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.m_Stream);
    Shape<2> sleft = GetShape(lhs.shape_, transpose_left);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    LOG_CHECK(dst.size(0) == sleft[0] && dst.size(1) == sright[1] && sleft[1] == sright[0])
    // use column major argument to compatible with most BLAS
    BLASEngine<xpu, DType>::gemm(
        dst.m_Stream, transpose_right, transpose_left, transpose_right ? rhs.size(0) : rhs.size(1),
        transpose_left ? lhs.size(1) : lhs.size(0), transpose_right ? rhs.size(1) : rhs.size(0),
        DType(scale * SV::AlphaBLAS()), rhs.__data_ptr, rhs.m_Stride, lhs.__data_ptr, lhs.m_Stride,
        DType(SV::BetaBLAS()), dst.__data_ptr, dst.m_Stride);
  }
};
template<typename SV, typename xpu, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 1, 1, 2, false, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 1, DType>* p_dst, const Tensor<xpu, 1, DType>& lhs,
                          const Tensor<xpu, 2, DType>& rhs, DType scale) {
    Tensor<xpu, 1, DType>& dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.m_Stream);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    LOG_CHECK(dst.size(0) == sright[1] && lhs.size(0) == sright[0])
    BLASEngine<xpu, DType>::gemv(dst.m_Stream, transpose_right, rhs.size(1), rhs.size(0),
                                 scale * SV::AlphaBLAS(), rhs.__data_ptr, rhs.m_Stride,
                                 lhs.__data_ptr, 1, SV::BetaBLAS(), dst.__data_ptr, 1);
  }
};
template<typename SV, typename xpu, typename DType>
struct DotEngine<SV, xpu, 2, 1, 1, true, false, DType> {
  inline static void Eval(Tensor<xpu, 2, DType>* p_dst, const Tensor<xpu, 1, DType>& lhs,
                          const Tensor<xpu, 1, DType>& rhs, DType scale) {
    Tensor<xpu, 2, DType>& dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.m_Stream);
    LOG_CHECK(dst.size(0) == lhs.size(0) && dst.size(1) == rhs.size(0))

    if (SV::BetaBLAS() == 0.0f) {
      BLASEngine<xpu, DType>::ger(dst.m_Stream, rhs.size(0), lhs.size(0), scale * SV::AlphaBLAS(),
                                  rhs.__data_ptr, 1, lhs.__data_ptr, 1, dst.__data_ptr,
                                  dst.m_Stride);
    } else {
      DotEngine<SV, xpu, 2, 2, 2, true, false, DType>::Eval(p_dst, lhs.Flatten2D(), rhs.Flatten2D(),
                                                            scale);
    }
  }
};

}  // namespace expr
}  // namespace mgloria

#endif