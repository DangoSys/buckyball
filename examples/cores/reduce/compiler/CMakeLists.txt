if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

set(BB_CORE_NAME reduce)
set(BB_CORE_TARGET_PREFIX "")
get_filename_component(_BB_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(_BB_F32ADD_DIALECT_DIR
  "${_BB_REPO_ROOT}/examples/balls/f32add/compiler/src/Dialect/Buckyball")
set(_BB_F32ADD_LEGALIZE "${_BB_F32ADD_DIALECT_DIR}/Transforms/LegalizeForLLVMExport.cpp")
if(NOT _BB_F32ADD_LEGALIZE IN_LIST BB_legalize_SOURCES)
  message(FATAL_ERROR "Reduce compiler requires the configured F32Add Ball lowering")
endif()
include_directories("${_BB_F32ADD_DIALECT_DIR}")

# The base builds the generated dialect and selected Ball IR/assign/export hooks.
add_subdirectory("${_BB_REPO_ROOT}/examples/cores/toy/compiler/src"
  "${CMAKE_CURRENT_BINARY_DIR}/base")
