/*!
 *@author   chenghua.wang
 *@file     vectorization/__vec_prepare.hpp
 *@brief    Used to generate the vectorization instruction code.
 * For CPU use, only. GPU don't need it, it has it's own way.
 */

#ifndef _MGLORIA___VEC_PREPARE_HPP_
#define _MGLORIA___VEC_PREPARE_HPP_
#include <malloc.h>
#include "../depends.hpp"
#include "../tensor.hpp"
#include "../expression.hpp"

namespace mgloria {
namespace vectorization {

enum class VecArch {
  NONE_Arch = 0,
  SSE_Arch = 1,
};

}  // namespace vectorization

}  // namespace mgloria

#endif