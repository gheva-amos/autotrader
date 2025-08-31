cmake_minimum_required(VERSION 3.20)
project(autotrader LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)

# Get the API code from IB and put it here:
set(IBAPI_DIR "${CMAKE_SOURCE_DIR}/IBJts" CACHE PATH "Path to IBJts")

if(NOT EXISTS "${IBAPI_DIR}/source/cppclient")
  message(FATAL_ERROR "Set IBAPI_DIR to the folder that contains source/cppclient (e.g., ~/SDK/IBJts)")
endif()

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS} ${GENERATED_DIR})
include_directories(include)


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
list(FILTER IBAPI_SOURCES EXCLUDE REGEX ".*/client/Decimal\\.cpp$")

add_library(ibapi STATIC ${IBAPI_SOURCES} ${PROTO_SRCS} ${PROTO_HDRS} ${CMAKE_SOURCE_DIR}/src/patch/decimal.cpp)
target_include_directories(ibapi PUBLIC
    "${IBAPI_DIR}/source/cppclient"
    "${IBAPI_DIR}/source/cppclient/client"
    "${IBAPI_DIR}/source/cppclient/shared"
    "${GENERATED_DIR}"
    )

if(UNIX)
  target_link_libraries(ibapi PUBLIC pthread protobuf)
endif()

add_subdirectory(experiments)
add_subdirectory(src)
