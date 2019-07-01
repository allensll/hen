cmake_minimum_required(VERSION 3.13)

project(hen CXX C)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

include(ExternalProject)

include(cmake/SEAL.cmake)

# External projects (seal)
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)





set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_SHARED "To build shared objects" OFF)


# find_package(SEAL 3.1.0 EXACT REQUIRED)
set(SEAL_HEADER ${EXTERNAL_INSTALL_INCLUDE_DIR})
set(SEAL_LIB ${EXTERNAL_INSTALL_LIB_DIR})

include_directories(${SEAL_HEADER})
link_directories(${SEAL_LIB})

# should do this?
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

