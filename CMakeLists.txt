cmake_minimum_required(VERSION 2.8)
project(gpu_benchmark)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

find_package(CUDA QUIET)
if(CUDA_FOUND)
  subdirs(src/gpu/)
endif()

