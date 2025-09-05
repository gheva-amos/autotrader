cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)

add_executable(connection_test connection_test.cpp)
target_compile_options(connection_test PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(connection_test PRIVATE ibapi)

add_executable(connection_raii connection_raii.cpp)
target_compile_options(connection_raii PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(connection_raii PRIVATE ibapi)

add_executable(contract_details contract_details.cpp)
target_compile_options(contract_details PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(contract_details PRIVATE autotrader_ib)

add_executable(market_data market_data.cpp)
target_compile_options(market_data PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(market_data PRIVATE autotrader_ib)

add_executable(zmq zmq.cpp)
target_compile_options(zmq PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter)
target_link_libraries(zmq PRIVATE libzmq cppzmq)
