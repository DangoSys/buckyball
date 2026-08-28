if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

# Rocket has no Ball lowering policy. Reuse the Ball-free compiler base rather
# than importing another Core's dialect or lowering implementation.
set(BB_CORE_NAME rocket)
set(BB_CORE_TARGET_PREFIX "")

get_filename_component(_BB_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
add_subdirectory("${_BB_REPO_ROOT}/examples/cores/toy/compiler/src"
  "${CMAKE_CURRENT_BINARY_DIR}/base")
