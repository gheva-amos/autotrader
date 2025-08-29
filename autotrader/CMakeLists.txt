cmake_minimum_required(VERSION 3.20)
project(autotrader LANGUAGES CXX)

set(IBAPI_DIR "${CMAKE_SOURCE_DIR}/IBJts" CACHE PATH "Path to IBJts")

if(NOT EXISTS "${IBAPI_DIR}/source/cppclient")
  message(FATAL_ERROR "Set IBAPI_DIR to the folder that contains source/cppclient (e.g., ~/SDK/IBJts)")
endif()

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS} ${GENERATED_DIR})

file(GLOB IBAPI_PROTO "${IBAPI_DIR}/source/proto/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
      PROTOS ${IBAPI_PROTO}
      PROTOC_OUT_DIR ${GENERATED_DIR}
    )

file(GLOB IBAPI_SOURCES
      "${IBAPI_DIR}/source/cppclient/client/*.cpp"
      "${IBAPI_DIR}/source/cppclient/shared/*.cpp"
      "${GENERATED_DIR}/*.cc"
    )

add_library(ibapi STATIC ${IBAPI_SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(ibapi PUBLIC
    "${IBAPI_DIR}/source/cppclient"
    "${IBAPI_DIR}/source/cppclient/client"
    "${IBAPI_DIR}/source/cppclient/shared"
    "${GENERATED_DIR}"
    )
if(UNIX)
  target_link_libraries(ibapi PUBLIC pthread)
endif()
