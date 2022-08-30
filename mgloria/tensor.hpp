/*!
 *@author   chenghua.wang
 *@file     tensor.hpp
 *@brief    The tensor class waiting for being inherited from GPU
 * and CPU implementations.
 *@note     m_xxx for member params, ms_xxx for static member params.
 */
#ifndef _MGLORIA_TENSOR_HPP_
#define _MGLORIA_TENSOR_HPP_
#pragma once
#include <iostream>
#include "depends.hpp"
#include "expression.hpp"
#include "tensor_shape.hpp"
#include "tensor_stream.hpp"

#if MGLORIA_USE_CUDA == 1
#include "stream_gpu.hpp"
#endif  // MGLORIA_USE_CUDA == 1

namespace mgloria {

/*!
 *@brief        Basic Init machine interface. Here is only the definition. The
 * implementation is in tensor_cpu.hpp and tensor_gpu.hpp .
 *@param        devIdx The device id. Figure out witch on should be initted.
 */
template<typename Device>
MGLORIA_INLINE_NORMAL void InitTensorComputeMachine(int devIdx);

/*!
 *@brief        Basic shutdown machine interface. Here is only the definition. The
 * implementation is in tensor_cpu.hpp and tensor_gpu.hpp .
 *@param        devIdx The device id. Figure out witch on should be shutdown.
 */
template<typename Device>
MGLORIA_INLINE_NORMAL void ShutdownTensorComputeMachine(int devIdx);

/*!
 *@brief        Choose the device for current thread to process.
 *@param        devIdx The device id. Figure out witch on should be choosen.
 */
template<typename Device>
MGLORIA_INLINE_NORMAL void SetCurrentDevice(int devIdx);

/*!
 *@brief        Tensor will be abstract as a expression named Tensor Expression
 * (TRValue) for the tensor is real value. The Container is the Tensor. and Dat-
 * aType is the tensor's data type.
 */
template<typename Container, typename Device, int Dims, typename DataType>
struct TRValue : public expr::RValueExpr<Container, DataType> {};

template<typename Device, int Dims, typename DataType = float>
class Tensor;

template<typename Device, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL std::ostream& operator<<(std::ostream& os,
                                               const Tensor<Device, Dims, DataType>& T);

//##############################################################################
//          Below is the definition of Tensor n dim and implementation         #
//##############################################################################

/*!
 *@brief        Tensor is the main data struct for u to use.
 * It works quite similar to numpy and other matrix based libs.
 *@example      The Tensor class can be easily created as code below.
 *              Tensor<CPU, 4> A; // The default type is float.
 *              Tensor<GPU, 3> B;
 * But, you should keep in mind that, the tensor class is just a abstract of data.
 * The DataType* __data_ptr maintained by this class is empty if you r not put it
 * to this class manually. Witch means, if you want your tensor has data, you should
 * follow the code bellow:
 *              Tensor<CPU, 4> A; A.SetShape(Shape<4>(2, 2, 2, 2)); A.SetData(new float[16]);
 *              Tensor<GPU, 4> B(new float[16], Shape<4>(2, 2, 2, 2));
 * If you want to use the tensor witch can allocate memory itself. You should use
 * `TensorT` class, this class will manage memory for you on both CPU and GPU;
 */
template<typename Device, int Dims, typename DataType>
class Tensor : public TRValue<Tensor<Device, Dims, DataType>, Device, Dims, DataType> {
 public:
  // ################### Friends #################################################
  template<typename out_Device, int out_Dims, typename out_DataType>
  friend std::ostream& operator<<(std::ostream& os,
                                  const Tensor<out_Device, out_Dims, out_DataType>& T);
  // ################### constructor impl ########################################
  MGLORIA_INLINE_NORMAL Tensor() {}

  MGLORIA_INLINE_NORMAL Tensor(const Shape<Dims>& shape) : m_Shape(shape) {}

  MGLORIA_INLINE_NORMAL Tensor(const Shape<Dims>& shape, Stream<Device>* stream)
      : m_Shape(shape), m_Stream(stream) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<Dims>& shape)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(shape[Dims - 1]) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<Dims>& shape, Stream<Device>* stream)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(shape[Dims - 1]), m_Stream(stream) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<Dims>& shape, index_t stride,
                               Stream<Device>* stream)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(stride), m_Stream(stream) {}

  // ################### parameters' definition and init #########################
  // device parameters.
  static const bool ms_UsingCPU = Device::usingCPU;
  static const bool ms_UsingGPU = Device::usingGPU;
  static const bool ms_AutoDevice = Device::AutoDevice;
  static const DeviceType m_DevType = Device::devType;

  // Shape and dims;
  index_t m_Stride_;
  static const int ms_Dimensions = Dims;
  static const int ms_Dimensions_in_use = Dims - 1;
  Shape<Dims> m_Shape;

  // Stream
  Stream<Device>* m_Stream = nullptr;

  // Data
  ///! Danger. For convience, it set to public but not named as a member params.
  DataType* __data_ptr = nullptr;

  // ################### Utils functions #########################################
  // Set and Get interface.
  MGLORIA_INLINE_NORMAL void SetStream(Stream<Device>* s) { m_Stream = s; }

  MGLORIA_INLINE_NORMAL Stream<Device>* GetStream() const { return m_Stream; }

  MGLORIA_INLINE_NORMAL void SetData(DataType* dptr) { __data_ptr = dptr; }

  MGLORIA_INLINE_NORMAL void SetShape(const Shape<Dims>& shape) { m_Shape = shape; }

  MGLORIA_INLINE_NORMAL const Shape<Dims>& GetShape() const { return m_Shape; }

  // Check memory Contiguous.
  MGLORIA_INLINE_NORMAL bool IsContiguous() const {
    return m_Shape[ms_Dimensions_in_use] == m_Stride_;
  }

  // Get element number.
  template<index_t start>
  MGLORIA_INLINE_NORMAL int32_t SubElementNum() const {
    int32_t ans = m_Stride_;
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(start, Dims);
    CHECK_GREATER_EQUAL(start, 0);
#endif
#pragma unroll
    for (int32_t i = start; i < ms_Dimensions_in_use; ++i) { ans *= m_Shape[i]; }
    return ans;
  }

  MGLORIA_INLINE_NORMAL int32_t AllElementNum() const { return SubElementNum<0>(); }

  // Get the size.
  MGLORIA_INLINE_NORMAL index_t size(index_t i) const { return m_Shape[i]; }

  // Get the Memory cost.
  template<index_t start>
  MGLORIA_INLINE_NORMAL size_t SubMemCost() const {
    return SubElementNum<start>() * sizeof(DataType);
  }

  MGLORIA_INLINE_NORMAL size_t AllMemCost() const { return SubElementNum<0>() * sizeof(DataType); }

  // Tensor shape operate.
  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType> Flatten1D() const {
    return Tensor<Device, 1, DataType>(__data_ptr, m_Shape.Flatten1D(), m_Stride_, m_Stream);
  }

  MGLORIA_INLINE_NORMAL Tensor<Device, 2, DataType> Flatten2D() const {
    return Tensor<Device, 2, DataType>(__data_ptr, m_Shape.Flatten2D(), m_Stride_, m_Stream);
  }

  /*!
   *@brief      slice the tensor in highest dimension [begin,end)
   *@param      start start position of slice
   *@param      end end position of slice
   *@return     Tensor<Device, Dims, DataType> after Sliced.
   *
   *@example    Tensor<CPU, 4, float> A = random(Shape<4>(6, 12, 1, 1));
   *            A = A.Slice(1, 2);
   *            std::cout << A.GetShape().str() << std::endl;
   * The out put is [Shape](1, 12, 1, 1). The highest dimension is cut.
   */
  MGLORIA_INLINE_NORMAL Tensor<Device, Dims, DataType> Slice(index_t start, index_t end) const {
    Shape<Dims> res = m_Shape;
    res[0] = end - start;
    return Tensor<Device, Dims, DataType>(__data_ptr + SubElementNum<1>() * start, res, m_Stride_,
                                          m_Stream);
  }

  // Operator overload
  /*!
   *@brief      Tensor operator =
   *@param      Tensor
   *@return     Tensor. Same size.
   *@example    Tensor<CPU, 4, float> A = random(Shape<4>(1, 3, 1, 1));
   *            Tensor<CPU, 4, float> B = A;
   * In this Tensor = Tensor situation. This function will be called.
   */
  MGLORIA_INLINE_NORMAL Tensor<Device, Dims, DataType>& operator=(
      const Tensor<Device, Dims, DataType>& T) {
    this->m_Shape = T.m_Shape;
    this->m_Stream = T.m_Stream;
    this->m_Stride_ = T.m_Stride_;
    this->__data_ptr = T.__data_ptr;
  }

  /*!
   *@brief      Tensor operator =
   *@param      expression Any kind of expression is OK.
   *@return     Tensor. Same size.
   *@example    Tensor<CPU, 3, float> A = random(Shape<4>(8, 8, 8));
   *            Tensor<CPU, 3, float> B = random(Shape<4>(8, 8, 8));
   *            Tensor<CPU, 3, float> C = random(Shape<4>(8, 8, 8));
   *            Tensor<CPU, 3, float> D = A + B + C;
   * In this situation. The A + B + C will be abstract to a expression.
   * When an expression need to save to a tensor, the compute behaviors
   * will be triggered.
   */
  template<typename SubType, expr::exprType EType>
  MGLORIA_INLINE_NORMAL Tensor<Device, Dims, DataType> operator=(
      const expr::Expression<SubType, DataType, EType>& expression) {
    return this->__dispatch(expression);
  }

  /*!
   *@brief      Tensor operator =
   *@param      scalar Any kind of scalar is OK.
   *@return     Tensor. Same size.
   *@example    Tensor<CPU, 3, float> A = random(Shape<3>(8, 8, 8));
   *            Tensor<CPU, 3, float> B = A * 3.f;
   *@note       Though DataType is a typename. But only the normal type can be put in.
   * which has already defined in data_type.hpp .
   */
  MGLORIA_INLINE_NORMAL Tensor<Device, Dims, DataType> operator=(const DataType& scalar) {
    return this->__dispatch(scalar);
  }

  /*!
   *@brief      Tensor operator []
   *@param      idx index_t
   *@return     Tensor dropped the highest dimension.
   *@example    Tensor<CPU, 4> A = random(Shape<4>(8, 8, 8, 8));
   *            Tensor<CPU, 3> B = A[0];
   *            Tensor<CPU, 2> C = B[0]; // Same as A[0][0];
   */
  MGLORIA_INLINE_NORMAL Tensor<Device, Dims - 1, DataType> operator[](index_t idx) {
    return Tensor<Device, Dims - 1, DataType>(__data_ptr + SubElementNum<1>() * idx,
                                              m_Shape.CudaShape(), m_Stride_, m_Stream);
  }
};  ///! Tensor greater than 1 dimension.

//##############################################################################
//          Below is the definition of Tensor 1 dim and implementation         #
//##############################################################################

/*!
 *@brief Tensor for only 1 dimensions. The function is same to Tensor
 * For n dimensions. You can check n dims Tensor for more information.
 */
template<typename Device, typename DataType>
class Tensor<Device, 1, DataType>
    : public TRValue<Tensor<Device, 1, DataType>, Device, 1, DataType> {
 public:
  // ################### constructor impl ########################################
  MGLORIA_INLINE_NORMAL Tensor() {}

  MGLORIA_INLINE_NORMAL Tensor(const Shape<1>& shape) : m_Shape(shape) {}

  MGLORIA_INLINE_NORMAL Tensor(const Shape<1>& shape, Stream<Device>* stream)
      : m_Shape(shape), m_Stream(stream) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<1>& shape)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(shape[0]) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<1>& shape, Stream<Device>* stream)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(shape[0]), m_Stream(stream) {}

  MGLORIA_INLINE_NORMAL Tensor(DataType* _dp, const Shape<1>& shape, index_t stride,
                               Stream<Device>* stream)
      : __data_ptr(_dp), m_Shape(shape), m_Stride_(stride), m_Stream(stream) {}

  // ################### parameters' definition and init #########################
  // device parameters.
  static const bool ms_UsingCPU = Device::usingCPU;
  static const bool ms_UsingGPU = Device::usingGPU;
  static const bool ms_AutoDevice = Device::AutoDevice;
  static const DeviceType m_DevType = Device::devType;
  // Shape and dims;
  index_t m_Stride_;
  Shape<1> m_Shape;
  // Stream
  Stream<Device>* m_Stream = nullptr;
  // Data
  ///! Danger. For convience, it set to public but not named as a member params.
  DataType* __data_ptr = nullptr;

  // ################### Utils functions #########################################
  // Set and Get interface.
  MGLORIA_INLINE_NORMAL void SetStream(Stream<Device>* s) { m_Stream = s; }

  MGLORIA_INLINE_NORMAL Stream<Device>* GetStream() const { return m_Stream; }

  MGLORIA_INLINE_NORMAL void SetData(DataType* dptr) { __data_ptr = dptr; }

  MGLORIA_INLINE_NORMAL void SetShape(const Shape<1>& shape) { m_Shape = shape; }

  MGLORIA_INLINE_NORMAL const Shape<1>& GetShape() const { return m_Shape; }

  // Check memory Contiguous.
  MGLORIA_INLINE_NORMAL bool IsContiguous() const { return true; }

  // Get element number.
  template<index_t start>  // still keep this template same as n-dimensions.
  MGLORIA_INLINE_NORMAL int32_t SubElementNum() const {
    return m_Shape[0];
  }

  MGLORIA_INLINE_NORMAL int32_t AllElementNum() const { return SubElementNum<0>(); }

  // Get the Memory cost.
  template<index_t start>
  MGLORIA_INLINE_NORMAL size_t SubMemCost() const {
    return SubElementNum<>() * sizeof(DataType);
  }

  MGLORIA_INLINE_NORMAL size_t AllMemCost() const { return SubElementNum<0>() * sizeof(DataType); }

  // Tensor shape operate.
  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType> Flatten1D() const {
    return Tensor<Device, 1, DataType>(__data_ptr, m_Shape.Flatten1D(), m_Stride_, m_Stream);
  }

  MGLORIA_INLINE_NORMAL Tensor<Device, 2, DataType> Flatten2D() const {
    return Tensor<Device, 2, DataType>(__data_ptr, m_Shape.Flatten2D(), m_Stride_, m_Stream);
  }

  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType> Slice(index_t start, index_t end) const {
    Shape<1> res = m_Shape;
    res[0] = end - start;
    return Tensor<Device, 1, DataType>(__data_ptr + start, res, res[0], m_Stream);
  }

  // Operator overload
  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType>& operator=(
      const Tensor<Device, 1, DataType>& T) {
    this->m_Shape = T.m_Shape;
    this->m_Stream = T.m_Stream;
    this->m_Stride_ = T.m_Stride_;
    this->__data_ptr = T.__data_ptr;
  }

  template<typename SubType, expr::exprType EType>
  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType> operator=(
      const expr::Expression<SubType, DataType, EType>& expression) {
    return this->__dispatch(expression);
  }

  MGLORIA_INLINE_NORMAL Tensor<Device, 1, DataType> operator=(const DataType& scalar) {
    return this->__dispatch(scalar);
  }

  MGLORIA_INLINE_NORMAL DataType& operator[](index_t idx) {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(idx, m_Shape[0]);
    CHECK_GREATER_EQUAL(idx, 0);
#endif
    return __data_ptr[idx];
  }

  MGLORIA_INLINE_NORMAL const DataType& operator[](index_t idx) const {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(idx, m_Shape[0]);
    CHECK_GREATER_EQUAL(idx, 0);
#endif
    return __data_ptr[idx];
  }

};  ///! tensor for only 1 dimension.

//##############################################################################
//          Below define the function needed to manage memory in TensorT       #
//##############################################################################

/*!
 *@brief        Allocate memory For Tensor.
 *@param        Tensor whose shape will be used to allocate memory.
 *@param        aligned If need to padding more space at dimension 0 to make
 * sure the last dimension aligned. If this flag set to true, more memory will
 * consumed by this  Tensor machine, while more efficiency.
 */
template<typename Device, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void MallocTensor(Tensor<Device, Dims, DataType>* t, bool aligned = false);

/*!
 *@brief        Free memory of Tensor.
 *@param        Tensor whose shape will be used to free memory.
 */
template<typename Device, int Dims, typename DataType>
MGLORIA_INLINE_NORMAL void FreeTensor(Tensor<Device, Dims, DataType>* t);

//##############################################################################
//          Below is the definition of TensorT and implementation              #
//##############################################################################

template<typename Device, int dims, typename DataType>
class TensorT : public Tensor<Device, dims, DataType> {
 public:
  TensorT(bool align = MGLORIA_PAD_TO_ALIGN) {}
  // TODO After Utils is implemented. This class could be implemented.
 public:
  bool align = false;
};  /// TensorT for more than 1 dims

//##############################################################################
//          Below is the definition of how to map the expression to tensor.    #
//          The implementation should in tensor_cpu.hpp and tensor_gpu.hpp .   #
//##############################################################################
/*!
 *@brief
 */
template<typename LV, typename RV, typename Device, int Dims, typename DataType, typename A_T,
         expr::exprType EType>
inline void MapExpr2Tensor(TRValue<RV, Device, Dims, DataType>* dst,
                           const expr::Expression<A_T, DataType, EType>& exp);

/*!
 *@brief
 */
template<typename LV, typename RV, int Dims, typename DataType, typename A_T, expr::exprType EType>
inline void MapExpr2Tensor(TRValue<RV, CPU, Dims, DataType>* dst,
                           const expr::Expression<A_T, DataType, EType>& exp);

/*!
 *@brief
 */
template<typename LV, typename RV, int Dims, typename DataType, typename A_T, expr::exprType EType>
inline void MapExpr2Tensor(TRValue<RV, GPU, Dims, DataType>* dst,
                           const expr::Expression<A_T, DataType, EType>& exp);

//##############################################################################
//          Below to define Tensor based operation such as Max, Min, SofMax    #
//          The implementation should in tensor_cpu.hpp and tensor_gpu.hpp .   #
//##############################################################################

}  // namespace mgloria

#endif