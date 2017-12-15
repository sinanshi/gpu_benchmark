include_directories(${CMAKE_SOURCE_DIR}/common/inc)
find_library(FOUND_CUBLAS cublas)

cuda_add_executable(bandwidth bandwidthTest.cu)
cuda_add_executable(simpleCUBLAS simpleCUBLAS.cpp)
cuda_add_executable(simpleCUFFT simpleCUFFT.cu)
cuda_add_cublas_to_target(simpleCUBLAS)
cuda_add_cufft_to_target(simpleCUFFT)
