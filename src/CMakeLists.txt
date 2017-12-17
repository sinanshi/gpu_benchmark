add_executable(test1 test.cpp)
target_link_libraries(test1 benchmark)
SET( CMAKE_CXX_FLAGS  "-O3" )
