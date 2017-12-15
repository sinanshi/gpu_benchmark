cmake_minimum_required(VERSION 2.8)
project(gpu_benchmark)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_files")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)



message(status ${CMAKE_MODULE_PATH})

find_package(CUDA QUIET)
#find_package(benchmark REQUIRED)







include(ExternalProject)

ExternalProject_Add(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git 
  GIT_TAG "master"
  PREFIX "${CMAKE_SOURCE_DIR}"
  DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/build/external/download
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/build/external/src"
  BINARY_DIR "${CMAKE_SOURCE_DIR}/build/external/bin"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
  #INSTALL_DIR "${CMAKE_SOURCE_DIR}/build/external/install"
  #TEST_BEFORE_INSTALL 1
  )


#include_directories(${PROJECT_BINARY_DIR}/3rdparty/bgen/3rd_party/zstd-1.1.0/lib)
#set(BGEN_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/3rdparty/bgen/genfile/include/)
#set(BGEN_LIBRARIES ${PROJECT_BINARY_DIR}/3rdparty/bgen/build/libbgen.a)
#set(ZSTD_LIBRARIES ${PROJECT_BINARY_DIR}/3rdparty/bgen/3rd_party/zstd-1.1.0/lib/libzstd.so)
#link_directories(${PROJECT_BINARY_DIR}/3rdparty/bgen/3rd_party/zstd-1.1.0/lib)
#include_directories(${BGEN_INCLUDE_DIRS})




#
#if(CUDA_FOUND)
#  subdirs(src/gpu/)
#endif()
#
#subdirs(src)
#
#
#

