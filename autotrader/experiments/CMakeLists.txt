cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)

add_executable(connection_test connection_test.cpp)
target_compile_options(connection_test PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(connection_test PRIVATE ibapi)
