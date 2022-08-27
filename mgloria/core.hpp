#ifndef _MGLORIA_CORE_HPP_
#define _MGLORIA_CORE_HPP_
#pragma once
#include "tensor_cpu.hpp"
#if MGLORIA_USE_CUDA == 1
#include "tensor_gpu.hpp"
#endif
#include "expr_eval.hpp"
#endif