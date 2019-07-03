# ${CMAKE_CURRENT_BINARY_DIR} is ??????

set(SEAL_PREFIX ${EXTERNAL_DIR}/seal)

ExternalProject_Add(seal
  PREFIX ${SEAL_PREFIX}
  GIT_REPOSITORY "https://github.com/Microsoft/SEAL.git"
  GIT_TAG "origin/3.3.0"
  # Skip updates
  UPDATE_COMMAND ""
  INSTALL_DIR ${EXTERNAL_DIR}
  SOURCE_SUBDIR native/src
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
  LOG_DOWNLOAD 1
  LOG_INSTALL 1
)

# ExternalProject_Get_Property(seal SOURCE_DIR)
add_library(libseal STATIC IMPORTED)
# set_target_properties(libseal PROPERTIES IMPORTED_LOCATION ${EXTERNAL_INSTALL_LIB_DIR}/libseal.a)
add_dependencies(libseal seal)