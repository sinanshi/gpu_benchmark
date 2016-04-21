come adfscmake_minimum_required(VERSION 2.8)
project(gpu_benchmark)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

find_package(CUDA QUIET REQUIRED)
subdirs(src/)

