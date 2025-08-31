cmake_minimum_required(VERSION 3.20)
project(autotrader LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)

add_library(autotrader_debug debug.cpp)
target_compile_options(autotrader_debug PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)

add_subdirectory(ib)
