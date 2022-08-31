/*!
 *@author   chenghua.wang
 *@file     vectorization/__vec_mm.hpp
 *@brief    The memory management for vectorization instruction operation.
 * Basically for SSE vectorization instruction set to use.
 */

#ifndef _MGLORIA___VEC_MM_HPP_
#define _MGLORIA___VEC_MM_HPP_

///! This file is include recursively. Conflict to __vec_prepare.hpp
#pragma once

#include <malloc.h>
#include "../tensor.hpp"
#include "./vectorization/__vec_prepare.hpp"

namespace mgloria {
namespace vectorization {

template<VecArch Arch>
MGLORIA_INLINE_NORMAL bool NotAlign(size_t pitch) {
  return !(pitch & ((1 << AlignBytes<Arch>::Default) - 1));
}

template<VecArch Arch>
MGLORIA_INLINE_NORMAL bool NotAlign(NO_TYPE_PTR ptr) {
  return NotAlign<Arch>(reinterpret_cast<size_t>(ptr));
}

template<VecArch Arch, typename DataType>
MGLORIA_INLINE_NORMAL index_t CeilAlign(index_t size) {
  const index_t aligned_bits = AlignBytes<Arch>::Default;
  const index_t masked = (1 << aligned_bits) - 1;
  const index_t data_size = sizeof(DataType);
  return (((size * data_size + masked) >> aligned_bits) << aligned_bits) / data_size;
}

template<VecArch Arch, typename DataType>
MGLORIA_INLINE_NORMAL index_t FloorAlign(index_t size) {
  const index_t aligned_bits = AlignBytes<Arch>::Default;
  const index_t data_size = sizeof(DataType);
  return (((size * data_size) >> aligned_bits) << aligned_bits) / data_size;
}

/*!
 *@brief        Work almost same as CUDA's cudaMallocPitch function. It will allocate a memory space
 *lines * line_cells cells.
 *@param        actual_mem the actual space allocate for each line.
 *@param        line_cells the cells required in each lines.
 *@param        lines the lines needed to allocate.
 *@details
 */
MGLORIA_INLINE_NORMAL NO_TYPE_PTR MallocAlignedPitch(size_t* actual_mem, size_t line_cells,
                                                     size_t lines) {
  const index_t aligned_bits = AlignBytes<MGLORIA_VECTORIZATION_ARCH>::Default;
  const index_t masked = (1 << aligned_bits) - 1;  // (1<<4) - 1 => 15
  size_t pitch_mem = ((line_cells + masked) >> aligned_bits) << aligned_bits;
  *actual_mem = pitch_mem;
  void* ans;
  int ret = posix_memalign(&ans, 1 << aligned_bits, pitch_mem * lines);
#if MGLORIA_CHECK_NULL_MEM_PTR == 1
  CHECK_EQUAL(ret, 0);
  if (ans == nullptr) {
    LOG_ERR << "CPU MAllocAlignedPitch Error! Want to allocate "
            << " lines=" << lines << ", line_cells=" << line_cells
            << ". The Bits Aligned is set to " << aligned_bits << ". The actual pitch mem is "
            << pitch_mem << std::endl;
    std::exit(MGLORIA_MEM_ERROR_EXIT);
  }
#endif  // MGLORIA_CHECK_NULL_MEM_PTR == 1
  return ans;
}

/*!
 *@brief        Free the aligned Pitch memory.
 *@param        ptr The data pointer returned by MallocAlignedPitch.
 */
MGLORIA_INLINE_NORMAL void FreeAlignedPitch(NO_TYPE_PTR ptr) { free(ptr); }

}  // namespace vectorization
}  // namespace mgloria

#endif  // _MGLORIA___VEC_MM_HPP_