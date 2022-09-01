/*!
 *@author   chenghua.wang
 *@file     tensor_shape.hpp
 *@brief    Define the shape basic template class for Tensor.
 *@note     Why use #pragma unroll ?
 * Because, the dims in loops is not ensured when program is compiled.
 * It's mutable when template is generated.
 */

#ifndef _MGLORIA_TENSOR_SHAPE_HPP_
#define _MGLORIA_TENSOR_SHAPE_HPP_
#pragma once
#include "depends.hpp"

namespace mgloria {

/*!
 *@brief    The basic shape template defined.
 */
template<int dims>
struct Shape;

/*!
 *@brief    The basic shape template implementation.
 */
template<int dims>
struct Shape {
  // #########################  constructor ########################
  ~Shape() = default;
  MGLORIA_INLINE_NORMAL Shape() {}
  MGLORIA_INLINE_NORMAL Shape(const Shape<dims>& s) {
    for (int32_t i = 0; i < dims; ++i) { _shape[i] = s[i]; }
  }

  // ############################ parameter contain. ################
  static const int32_t dimensions = dims;
  static const int32_t dimensions_in_use = dims - 1;
  index_t _shape[dimensions];

  // ######################### operator overload ####################
  MGLORIA_INLINE_NORMAL index_t& operator[](index_t i) {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(i, dims, " ", i, "is out of Bound for dim=", dims);
    CHECK_GREATER_EQUAL(i, 0, " ", i, "is out of Bound for dim=", dims);
#endif
    return _shape[i];
  }

  MGLORIA_INLINE_NORMAL const index_t& operator[](index_t i) const {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(i, dims, " ", i, "is out of Bound for dim=", dims);
    CHECK_GREATER_EQUAL(i, 0, " ", i, "is out of Bound for dim=", dims);
#endif
    return _shape[i];
  }

  MGLORIA_INLINE_NORMAL bool operator==(const Shape<dims>& s) const {
#pragma unroll
    for (uint32_t i = 0; i < dims; ++i) {
      if (_shape[i] != s[i]) return false;
    }
    return true;
  }

  MGLORIA_INLINE_NORMAL bool operator!=(const Shape<dims>& s) const { return !(*this == s); }

  // ############################# Utils functions #################

  /*!
   *@brief      Return the print string.
   *@return     std::string.
   *@note       There is no performance consider for this function.
   * The print function is not used in actual compute period, right?
   */
  MGLORIA_INLINE_NORMAL std::string str() const {
    std::stringstream s;
    s << "[Shape](";
#pragma unroll
    for (index_t i = 0; i < dims - 1; ++i) { s << std::to_string(_shape[i]) << ", "; }
    s << std::to_string(_shape[dimensions_in_use]) << ")\n";
    return s.str();
  }

  /*!
   *@brief      Return rhe elements one tensor have.
   *@return     index_t number of elements.
   */
  MGLORIA_INLINE_NORMAL index_t Size() const {
    index_t ans = 1;
#pragma unroll
    for (int32_t i = 0; i < dims; ++i) { ans *= _shape[i]; }
    return ans;
  }

  /*!
   *@brief      Get the Size of element in [start_dim, end_dim)
   *@param      start_dim the start dim you need to index.
   *@param      end_dim the end dim you need to index.
   *@return     index_t the Sub Size of Tensor's Element.
   */
  MGLORIA_INLINE_NORMAL index_t SubSize(index_t start_dim, index_t end_dim) const {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_LOWER_THAN(start_dim, dims, " ", start_dim, "is out of Bound for dim=", dims);
    CHECK_GREATER_EQUAL(start_dim, 0, " ", start_dim, "is out of Bound for dim=", dims);
    CHECK_LOWER_THAN(end_dim, dims, " ", end_dim, "is out of Bound for dim=", dims);
    CHECK_GREATER_EQUAL(end_dim, 0, " ", end_dim, "is out of Bound for dim=", dims);
#endif
    index_t ans = 1;
#pragma unroll
    for (index_t i = start_dim; i < end_dim; ++i) { ans *= _shape[i]; }
    return ans;
  }

  /*!
   *@brief      Get the Shape between dim [s, e)
   *@param      s start dim.
   *@param      e end dim.
   *@return     Shape<s - e> the shape between dim s and e.
   */
  template<index_t s, index_t e>
  MGLORIA_INLINE_NORMAL Shape<e - s> Slice() const {
#if MGLORIA_ARRAY_BOUND_CHECK == 1
    CHECK_GREATER_THAN(e, s, " ", e, " need to be lager then ", s);
#endif
    Shape<e - s> ans;
#pragma unroll
    for (index_t i = s; i < e; ++i) { ans[i - s] = _shape[i]; }
    return ans;
  }

  /*!
   *@brief      Flatten the nd Tensor to 1d Tensor.
   *@return     Shape<1> one dim Shape.
   *@details    The final one dim is the result of multiplication
   * of nd Tensor's each dim.
   */
  MGLORIA_INLINE_NORMAL Shape<1> Flatten1D() const {
    Shape<1> ans;
    ans[0] = Size();
    return ans;
  }

  /*!
   *@brief      Flatten the nd Tensor to 2d Tensor.
   *@return     Shape<2> two dims Shape.
   *@details    The Shape<2>'s second dimension is the highest dim of nd Tensor.
   * Other dims in nd Tensor will be multiplied, and the result is the the first
   * dim of Shape<2>.
   */
  MGLORIA_INLINE_NORMAL Shape<2> Flatten2D() const {
    Shape<2> ans;
    ans[1] = _shape[dimensions_in_use];
    int32_t tmp = 1;
#pragma unroll
    for (int32_t i = 0; i < dimensions_in_use; ++i) { tmp *= _shape[i]; }
    ans[0] = tmp;
    return ans;
  }

  /*!
   *@brief      Take off the largest dims for cuda to use.
   *@return     Shape<dimensions_in_use>
   *@note       For CUDA use only.
   */
  MGLORIA_INLINE_NORMAL Shape<dimensions_in_use> CudaShape() const {
    Shape<dimensions_in_use> ans;
#pragma unroll
    for (index_t i = 0; i < dimensions_in_use; ++i) { ans[i] = _shape[i + 1]; }
    return ans;
  }
};

//######################### Some functions make Shapes #######################

/*!
 *@brief        make Shape 1 dim.
 *@param        d_1 the shape of dim 1
 *@return       Shape<1>
 */
MGLORIA_INLINE_NORMAL Shape<1> makeShape1d(index_t d_1) {
  Shape<1> ans;
  ans[0] = d_1;
  return ans;
}

/*!
 *@brief        make Shape 2 dim.
 *@param        d_1 the shape of dim 1
 *@param        d_2 the shape of dim 2
 *@return       Shape<2>
 */
MGLORIA_INLINE_NORMAL Shape<2> makeShape2d(index_t d_1, index_t d_2) {
  Shape<2> ans;
  ans[0] = d_1, ans[1] = d_2;
  return ans;
}

/*!
 *@brief        make Shape 3 dim.
 *@param        d_1 the shape of dim 1
 *@param        d_2 the shape of dim 2
 *@param        d_3 the shape of dim 3
 *@return       Shape<3>
 */
MGLORIA_INLINE_NORMAL Shape<3> makeShape3d(index_t d_1, index_t d_2, index_t d_3) {
  Shape<3> ans;
  ans[0] = d_1, ans[1] = d_2, ans[2] = d_3;
  return ans;
}

/*!
 *@brief        make Shape 4 dim.
 *@param        d_1 the shape of dim 1
 *@param        d_2 the shape of dim 2
 *@param        d_3 the shape of dim 3
 *@param        d_4 the shape of dim 4
 *@return       Shape<4>
 */
MGLORIA_INLINE_NORMAL Shape<4> makeShape4d(index_t d_1, index_t d_2, index_t d_3, index_t d_4) {
  Shape<4> ans;
  ans[0] = d_1, ans[1] = d_2, ans[2] = d_3, ans[3] = d_4;
  return ans;
}

/*!
 *@brief        make Shape 5 dim.
 *@param        d_1 the shape of dim 1
 *@param        d_2 the shape of dim 2
 *@param        d_3 the shape of dim 3
 *@param        d_4 the shape of dim 4
 *@param        d_5 the shape of dim 5
 *@return       Shape<5>
 */
MGLORIA_INLINE_NORMAL Shape<5> makeShape5d(index_t d_1, index_t d_2, index_t d_3, index_t d_4,
                                           index_t d_5) {
  Shape<5> ans;
  ans[0] = d_1, ans[1] = d_2, ans[2] = d_3, ans[3] = d_4, ans[4] = d_5;
  return ans;
}

/*!
 *@brief    Convert Shape<3> from src layout to dist.
 *@param    rhs Shape<3> pass by reference.
 *@param    src rhs's Layout
 *@param    dist The Layout you want to convert to.
 *@return   Shape<3>
 */
MGLORIA_INLINE_NORMAL Shape<3> ConvertLayout(const Shape<3>& rhs, const LayoutTypeType& src,
                                             const LayoutTypeType& dist) {
  Shape<3> ans;
  ans[0] = -1, ans[1] = -1, ans[2] = -1;
  // Normalized the src Layout to HWC.
  switch (src) {
    case LayoutTypeType::CHW: {
      ans[0] = rhs[1];
      ans[1] = rhs[2];
      ans[2] = rhs[0];
      break;
    }
    case LayoutTypeType::HWC: {
      ans = rhs;
      break;
    }
    default: {
      LOG_ERR << "There is no 3 dimensions Layout as you gave.\n";
      break;
    }
  }
  // Refine the Layout to dist.
  switch (dist) {
    case LayoutTypeType::HWC: {
      return ans;
    }
    case LayoutTypeType::CHW: {
      Shape<3> tmp;
      tmp[0] = ans[2];
      tmp[1] = ans[0];
      tmp[2] = ans[1];
      return tmp;
    }
    default: {
      LOG_ERR << "There is no 3 dimensions Layout as you gave.\n";
      break;
    }
  }
  return ans;
}

/*!
 *@brief    Convert Shape<4> from src layout to dist.
 *@param    rhs Shape<4> pass by reference.
 *@param    src rhs's Layout
 *@param    dist The Layout you want to convert to.
 *@return   Shape<4>
 */
MGLORIA_INLINE_NORMAL Shape<4> ConvertLayout(const Shape<4>& rhs, const LayoutTypeType& src,
                                             const LayoutTypeType& dist) {
  Shape<4> ans, tmp;
  // Normalized the src Layout to BCHW.
  switch (src) {
    case LayoutTypeType::BCHW: {
      ans = rhs;
      break;
    }
    case LayoutTypeType::BHWC: {
      ans[0] = rhs[0];
      ans[1] = rhs[3];
      ans[2] = rhs[1];
      ans[3] = rhs[2];
      break;
    }
    case LayoutTypeType::CHWB: {
      ans[0] = rhs[3];
      ans[1] = rhs[0];
      ans[2] = rhs[1];
      ans[3] = rhs[2];
      break;
    }
    default: {
      LOG_ERR << "There is no 4 dimensions Layout as you gave.\n";
      break;
    }
  }
  // Refine the Layout to dist.
  switch (dist) {
    case LayoutTypeType::BCHW: {
      return ans;
    }
    case LayoutTypeType::BHWC: {
      tmp[0] = ans[0];
      tmp[1] = ans[2];
      tmp[2] = ans[3];
      tmp[3] = ans[1];
      return tmp;
    }
    case LayoutTypeType::CHWB: {
      tmp[0] = ans[1];
      tmp[1] = ans[2];
      tmp[2] = ans[3];
      tmp[3] = ans[0];
      return tmp;
    }
    default: {
      LOG_ERR << "There is no 4 dimensions Layout as you gave.\n";
      break;
    }
  }
  return ans;
}

/*!
 *@brief    Convert Shape<5> from src layout to dist.
 *@param    rhs Shape<5> pass by reference.
 *@param    src rhs's Layout
 *@param    dist The Layout you want to convert to.
 *@return   Shape<5>
 */
MGLORIA_INLINE_NORMAL Shape<5> ConvertLayout(const Shape<5>& rhs, const LayoutTypeType& src,
                                             const LayoutTypeType& dist) {
  Shape<5> ans, tmp;
  // Normalized the src Layout to BCDHW.
  switch (src) {
    case LayoutTypeType::BCDHW: {
      ans = rhs;
      break;
    }
    case LayoutTypeType::BDHWC: {
      ans[0] = rhs[0];
      ans[1] = rhs[4];
      ans[2] = rhs[1];
      ans[3] = rhs[2];
      ans[4] = rhs[3];
      break;
    }
    case LayoutTypeType::CDHWB: {
      ans[0] = rhs[4];
      ans[1] = rhs[0];
      ans[2] = rhs[1];
      ans[3] = rhs[2];
      ans[4] = rhs[3];
      break;
    }
    default: {
      LOG_ERR << "There is no 5 dimensions Layout as you gave.\n";
      break;
    }
  }
  // Refine the Layout to dist.
  switch (dist) {
    case LayoutTypeType::BCDHW: {
      return ans;
    }
    case LayoutTypeType::BDHWC: {
      tmp[0] = ans[0];
      tmp[1] = ans[2];
      tmp[2] = ans[3];
      tmp[3] = ans[4];
      tmp[4] = ans[1];
      return tmp;
    }
    case LayoutTypeType::CDHWB: {
      tmp[0] = ans[1];
      tmp[1] = ans[2];
      tmp[2] = ans[3];
      tmp[3] = ans[4];
      tmp[4] = ans[0];
      return tmp;
    }
    default: {
      LOG_ERR << "There is no 5 dimensions Layout as you gave.\n";
      break;
    }
  }
  return ans;
}

}  // namespace mgloria

#endif