if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

set(BB_CORE_NAME decode)
set(BB_CORE_TARGET_PREFIX "")

set(BB_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
get_filename_component(_BB_REPO_ROOT "${BB_CORE_DIR}/../../.." ABSOLUTE)
include_directories(${_BB_REPO_ROOT}/examples/balls)
add_subdirectory(src)
