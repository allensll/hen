cmake_minimum_required(VERSION 3.13)

project(HENExamples)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)

# add_executable(example1 example1.cpp)
add_executable(example3 example3.cpp)

# Link hen
set(HEN_LIB ${CMAKE_SOURCE_DIR}/../lib)
set(HEN_HEADER ${CMAKE_SOURCE_DIR}/../include)
# link_directories(${HEN_LIB})
# target_link_libraries(example1 ${HEN_LIB}/libhen.a)
include_directories(${HEN_HEADER})

# Import SEAL
find_package(SEAL 3.1.0 EXACT REQUIRED)

# Link SEAL
# target_link_libraries(example1 ${HEN_LIB}/libhen.a SEAL::seal)
target_link_libraries(example3 ${HEN_LIB}/libhen.a SEAL::seal stdc++fs)
