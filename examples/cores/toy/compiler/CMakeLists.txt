if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

set(BB_CORE_NAME toy)
set(BB_CORE_TARGET_PREFIX "")

add_subdirectory(src)
