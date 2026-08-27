if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()
if(NOT DEFINED BUCKYBALL_CHIP_PB OR BUCKYBALL_CHIP_PB STREQUAL "")
  message(FATAL_ERROR "BUCKYBALL_CHIP_PB is required")
endif()
if(NOT EXISTS "${BUCKYBALL_CHIP_PB}")
  message(FATAL_ERROR "missing Chip.pb: ${BUCKYBALL_CHIP_PB}")
endif()

get_filename_component(BB_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(BB_COMPILER_COMMON_DIR "${BB_REPO_ROOT}/compiler/src")
set(BB_TARGET_REGISTRY_INC "${CMAKE_CURRENT_BINARY_DIR}/BuckyballTargetRegistry.inc")
set(BB_TARGET_REGISTRY_SOURCE
  "${BB_COMPILER_COMMON_DIR}/Target/BuckyballTargetRegistry.cpp")
set(BB_TARGET_REGISTRY_GENERATOR
  "${BB_REPO_ROOT}/compiler/scripts/pb_to_target_registry.py")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --out "${BB_TARGET_REGISTRY_INC}"
  RESULT_VARIABLE BB_TARGET_REGISTRY_RESULT
  OUTPUT_VARIABLE BB_TARGET_REGISTRY_OUTPUT
  ERROR_VARIABLE BB_TARGET_REGISTRY_ERROR)
if(NOT BB_TARGET_REGISTRY_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to generate Buckyball target registry:\n${BB_TARGET_REGISTRY_OUTPUT}${BB_TARGET_REGISTRY_ERROR}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${BUCKYBALL_CHIP_PB}" "${BB_TARGET_REGISTRY_GENERATOR}")

include_directories("${BB_COMPILER_COMMON_DIR}")
include_directories("${BB_COMPILER_COMMON_DIR}/Dialect")
include_directories("${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-core-compiler-dirs
  RESULT_VARIABLE BB_CORE_COMPILER_DIR_RESULT
  OUTPUT_VARIABLE BB_CORE_COMPILER_DIRS
  ERROR_VARIABLE BB_CORE_COMPILER_DIR_ERROR)
if(NOT BB_CORE_COMPILER_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to resolve Core compiler package:\n${BB_CORE_COMPILER_DIR_ERROR}")
endif()
string(STRIP "${BB_CORE_COMPILER_DIRS}" BB_CORE_COMPILER_DIRS)
string(REPLACE "\n" ";" BB_CORE_COMPILER_DIRS "${BB_CORE_COMPILER_DIRS}")
list(LENGTH BB_CORE_COMPILER_DIRS BB_CORE_COMPILER_DIR_COUNT)
if(NOT BB_CORE_COMPILER_DIR_COUNT EQUAL 1)
  message(FATAL_ERROR
    "a compiler build must select exactly one Core package; got: ${BB_CORE_COMPILER_DIRS}")
endif()
list(GET BB_CORE_COMPILER_DIRS 0 BB_CORE_COMPILER_DIR)
get_filename_component(BB_CORE_COMPILER_NAME "${BB_CORE_COMPILER_DIR}" NAME)
add_subdirectory("${BB_CORE_COMPILER_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/cores/${BB_CORE_COMPILER_NAME}")
