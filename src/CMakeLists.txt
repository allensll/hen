file(GLOB HEN_SRC_FILES RELATIVE "${CMAKE_SOURCE_DIR}/src" "*.cpp")
file(GLOB HEN_SRC_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/src" "*.h")

if (BUILD_SHARED)
add_library(hen SHARED ${HEN_SRC_FILES})
else (BUILD_SHARED)
add_library(hen STATIC ${HEN_SRC_FILES})
endif (BUILD_SHARED)

add_custom_command(
  TARGET hen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/include
  COMMAND ${CMAKE_COMMAND} -E copy
    ${HEN_SRC_HEADERS}
    ${CMAKE_SOURCE_DIR}/include)
