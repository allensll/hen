# ${CMAKE_CURRENT_BINARY_DIR} is ??????

set(SEAL_PREFIX ${CMAKE_BINARY_DIR}/external/seal)

ExternalProject_Add(seal
  PREFIX ${SEAL_PREFIX}
  GIT_REPOSITORY "https://github.com/Microsoft/SEAL.git"
  GIT_TAG "3.3.0"
  # Skip updates
  UPDATE_COMMAND ""
  INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
  CONFIGURE_COMMAND cmake ${SEAL_PREFIX}/src/seal/native/src
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
             -DCMAKE_CXX_FLAGS=${SEAL_CXX_FLAGS}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  LOG_DOWNLOAD 1
  LOG_INSTALL 1
)

ExternalProject_Get_Property(seal SOURCE_DIR)
add_library(libseal STATIC IMPORTED)
set_target_properties(libseal PROPERTIES IMPORTED_LOCATION ${EXTERNAL_INSTALL_LIB_DIR}/libseal.a)
add_dependencies(libseal seal)