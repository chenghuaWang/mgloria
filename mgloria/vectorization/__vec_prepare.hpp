/*!
 *@author   chenghua.wang
 *@file     vectorization/__vec_prepare.hpp
 *@brief    Used to generate the vectorization instruction code.
 * For CPU use, only. GPU don't need it, it has it's own way.
 */

#ifndef _MGLORIA___VEC_PREPARE_HPP_
#define _MGLORIA___VEC_PREPARE_HPP_

#pragma once

#include "../depends.hpp"
#include "../tensor.hpp"
#include "../expression.hpp"

namespace mgloria {
namespace vectorization {

enum class VecArch {
  NONE_Arch = 0,
  SSE_Arch = 1,
};

/*!
 *@brief      Quite similar to cub::AlignBytes in cuda toolkit.
 */
template<VecArch ArchType>
struct AlignBytes {
  static const index_t Default = MGLORIA_DEFAULT_ALIGNBYTES;
};

}  // namespace vectorization
}  // namespace mgloria

#include "./vectorization/__vec_mm.hpp"

namespace mgloria {
namespace vectorization {

template<typename DataType, VecArch Arch = MGLORIA_VECTORIZATION_ARCH>
struct Vectorized {};

template<typename OP, typename DataType, VecArch Arch>
struct VectorizedOP {
  static const bool m_Enable = MGLORIA_VECTORIZATION_FALSE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_plus, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return lhs + rhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_minus, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return lhs - rhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_mul, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return lhs * rhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_div, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return lhs / rhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_left, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return lhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_right, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(const Vectorized<DataType, Arch>& lhs,
                                                          const Vectorized<DataType, Arch>& rhs) {
    return rhs;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename DataType, VecArch Arch>
struct VectorizedOP<op::_identity, DataType, Arch> {
  MGLORIA_INLINE_CPU static Vectorized<DataType, Arch> Do(
      const Vectorized<DataType, Arch>& single) {
    return single;
  }

  static const bool m_Enable = MGLORIA_VECTORIZATION_TRUE;
};

template<typename LeftValue, typename TFloat, VecArch Arch>
struct VectorizedSaver {
  MGLORIA_INLINE_CPU static void Save(TFloat* dst, const Vectorized<TFloat, Arch>& src) {
    Vectorized<TFloat, Arch> lhs = Vectorized<TFloat, Arch>::Load(dst);
    Vectorized<TFloat, Arch> ans =
        VectorizedOP<typename LeftValue::OPType, TFloat, Arch>::Do(lhs, src);
    ans.Store(dst);
  }
};
template<typename TFloat, VecArch Arch>
struct VectorizedSaver<op::_saveto, TFloat, Arch> {
  MGLORIA_INLINE_CPU static void Save(TFloat* dst, const Vectorized<TFloat, Arch>& src) {
    src.Store(dst);
  }
};

}  // namespace vectorization
}  // namespace mgloria

#endif