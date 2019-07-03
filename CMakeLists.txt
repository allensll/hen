cmake_minimum_required(VERSION 3.13)

project(hen CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

# External projects (seal)
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_DIR}/include)

include(ExternalProject)

include(cmake/SEAL.cmake)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_library(libseal )

set(HEN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/hen)

add_library(hen)
add_dependencies(hen seal)

target_include_directories(hen

)

target_link_libraries(hen
    seal
)


# option(BUILD_SHARED "To build shared objects" OFF)


# find_package(SEAL 3.3.0 EXACT REQUIRED)
# set(SEAL_HEADER ${EXTERNAL_INSTALL_INCLUDE_DIR})
# set(SEAL_LIB ${EXTERNAL_INSTALL_LIB_DIR})

# include_directories(${SEAL_HEADER})
# link_directories(${SEAL_LIB})

# # should do this?
# set(CMAKE_CXX_STANDARD 17)

# add_subdirectory(src)

