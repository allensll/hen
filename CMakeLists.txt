cmake_minimum_required(VERSION 3.13)

project(hen)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_SHARED "To build shared objects" OFF)

# Import SEAL
# find_package(SEAL 3.1.0 EXACT REQUIRED)
set(SEAL_HEADER /usr/local/include)
set(SEAL_LIB /usr/local/lib)

include_directories(${SEAL_HEADER})
link_directories(${SEAL_LIB})

# should do this?
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

