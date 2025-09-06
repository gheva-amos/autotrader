cmake_minimum_required(VERSION 3.20)
project(autotrader LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)

add_executable(autotrader
    base_thread.cpp
    router.cpp
    pull.cpp
    pub.cpp
    autotrader.cpp
    )
target_compile_options(autotrader PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(autotrader PUBLIC autotrader_ib libzmq cppzmq)
