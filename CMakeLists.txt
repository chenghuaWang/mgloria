#[[
@author     chenghua.wang
@startdate  Aug 22, 2022
@file       CMakeLists.txt
@brief      The main CMakeLists file for organizing all project.
@details    Some options are provided when you want to rebuild this
project, the options are shown below:
1. 
]]
cmake_minimum_required(VERSION 3.22)

set(CMAKE_INSTALL_MESSAGE LAZY CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

option(BUILD_THIRD_PARTY "Build the third_party." OFF)
option(BUILD_MGLORIA "Build the matrix part of gloria" ON)

if (NOT BUILD_THIRD_PARTY AND NOT BUILD_MGLORIA)
    message(FATAL_ERROR "At least one target should be setted. [third_party] or [mgloria]")
endif()

option(USE_CLANG_FORMAT "" ON)
option(USE_CLANG_TIDY "" ON)
option(USE_CUDA "using NVIDIA Graphic Cards supported for cuda to sccelerate." ON)
option(BUILD_TESTING "" OFF)
# Reference:
# https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option(OF_FORCE_COLORED_DIAGNOSTICS "Always produce ANSI-colored diagnostics (GNU/Clang only)." ON)

if(APPLE)
  set(BUILD_CUDA OFF)
endif()

set(CPU_THREADING_RUNTIME "OMP")

set(COMPILER_VERSION_ERROR_MSG "At least gcc 9, clang 12 supported.")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 9)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
  if(CPU_THREADING_RUNTIME STREQUAL "OMP")
    set(OMP_FLAGS "-fopenmp")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Reference:
  # https://releases.llvm.org/11.0.0/tools/clang/docs/OpenMPSupport.html
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 11)
    if(CPU_THREADING_RUNTIME STREQUAL "OMP")
      message(
        FATAL_ERROR
          "libopenmp is not supported under clang10, please use TBB with '-DCPU_THREADING_RUNTIME=TBB'."
      )
    else()
        set(OMP_FLAGS "-fopenmp")
    endif()
  endif()
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 5)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 12)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
else()
  message(WARNING "Unknown compiler \"${CMAKE_CXX_COMPILER_ID}\".")
endif()

set(mgloria_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)
get_filename_component(real_src_dir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(real_bin_dir "${CMAKE_BINARY_DIR}" REALPATH)

list(APPEND CMAKE_MODULE_PATH ${mgloria_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${mgloria_cmake_dir})

if(CPU_THREADING_RUNTIME STREQUAL "TBB")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_TBB)
elseif(CPU_THREADING_RUNTIME STREQUAL "OMP")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_OMP)
elseif(CPU_THREADING_RUNTIME STREQUAL "SEQ")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_SEQ)
else()
  message(FATAL_ERROR "CPU_THREADING_RUNTIME must be one of: TBB, OMP, SEQ")
endif()

if(OF_FORCE_COLORED_DIAGNOSTICS)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics>>)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
    CACHE PATH "Where to install third party headers and libs")

if (USE_CUDA)
    include("${mgloria_cmake_dir}/CUDA.cmake")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)
        # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
        list(APPEND CMAKE_CUDA_ARCHITECTURES 61-real)
        # V100, TITAN V
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)
        if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
            # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
        endif()
        if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
            # A100
            list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
        endif()
        if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
            # GeForce RTX 30xx
            list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
        endif()
    endif()

    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_SEPARABLE_COMPILATION OFF)

    if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.7")
            set(CUDA_NVCC_THREADS_NUMBER "4" CACHE STRING "")
            list(APPEND CUDA_NVCC_FLAGS -t ${CUDA_NVCC_THREADS_NUMBER})
        endif()
        message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
        list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
    endif()
endif()

message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

set(mgloria_head_dir ${PROJECT_SOURCE_DIR}/mgloria)
set(mgloria_ps_head_dir ${PROJECT_SOURCE_DIR}/mgloria_ps)

file(GLOB_RECURSE mgloria_files *.hpp *.cuh)

add_subdirectory(mgloria)
add_subdirectory(mgloria_ps)
add_subdirectory(test)
