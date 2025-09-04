cmake_minimum_required(VERSION 3.20)
project(autotrader LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)

add_library(autotrader_ib
    ib_client.cpp
    market_data.cpp
    )
target_compile_options(autotrader_ib PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(autotrader_ib PUBLIC ibapi autotrader_debug)
