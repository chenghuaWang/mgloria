/*!
 *@author   chenghua.wang
 *@file     data_layout.hpp
 *@brief    The layout of data.
 *@note     Support 3dim, 4dim(default), 5dim in different sequential.
 */

#ifndef _MGLORIA_DATA_LAYOUT_HPP
#define _MGLORIA_DATA_LAYOUT_HPP
#pragma once
#include "prepare.hpp"

namespace mgloria {

/*!
 *@brief The BCHW, BCDHW, etc kinds of things is inspired by cuDNN.
 */
enum class LayoutTypeType : uint8_t {
  // Normal 4dim like tensor in torch. Batch, Channel, Height, Wight.
  BCHW = 0,
  BHWC = 1,
  CHWB = 2,
  // 3dim Tensor.
  HWC = 3,
  CHW = 4,
  // 5dims
  BCDHW = 5,
  BDHWC = 6,
  CDHWB = 7
};

const LayoutTypeType defualt_layout_t = LayoutTypeType::BCHW;      ///! 4dims BCHW is default.
const LayoutTypeType defualt_layout_5d_t = LayoutTypeType::BDHWC;  ///! 5dims default.

template<LayoutTypeType T>
struct LayoutType;

template<>
struct LayoutType<LayoutTypeType::BCHW> {
  static const int32_t Dims = 4;
  // TODO add cuDNN format and flags.
};

template<>
struct LayoutType<LayoutTypeType::BHWC> {
  static const int32_t Dims = 4;
};

template<>
struct LayoutType<LayoutTypeType::CHWB> {
  static const int32_t Dims = 4;
};

template<>
struct LayoutType<LayoutTypeType::HWC> {
  static const int32_t Dims = 3;
};

template<>
struct LayoutType<LayoutTypeType::CHW> {
  static const int32_t Dims = 3;
};

template<>
struct LayoutType<LayoutTypeType::BCDHW> {
  static const int32_t Dims = 5;
};

template<>
struct LayoutType<LayoutTypeType::BDHWC> {
  static const int32_t Dims = 5;
};

template<>
struct LayoutType<LayoutTypeType::CDHWB> {
  static const int32_t Dims = 5;
};

}  // namespace mgloria
#endif