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
set(BB_TARGET_DEFINITIONS "${CMAKE_CURRENT_BINARY_DIR}/BuckyballAll.td")
set(BB_BALL_LOWERING_HOOKS "${CMAKE_CURRENT_BINARY_DIR}/BuckyballBallLoweringHooks.inc")
set(BB_TARGET_REGISTRY_GENERATOR
  "${BB_REPO_ROOT}/compiler/scripts/pb_to_target_registry.py")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --out "${BB_TARGET_REGISTRY_INC}"
    --td-out "${BB_TARGET_DEFINITIONS}"
    --lowering-hooks-out "${BB_BALL_LOWERING_HOOKS}"
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
    --print-ball-compiler-paths dialect-dir
  RESULT_VARIABLE BB_BALL_DIALECT_DIR_RESULT
  OUTPUT_VARIABLE BB_BALL_DIALECT_DIRS
  ERROR_VARIABLE BB_BALL_DIALECT_DIR_ERROR)
if(NOT BB_BALL_DIALECT_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to resolve Ball compiler dialect directories:\n${BB_BALL_DIALECT_DIR_ERROR}")
endif()
string(STRIP "${BB_BALL_DIALECT_DIRS}" BB_BALL_DIALECT_DIRS)
string(REPLACE "\n" ";" BB_BALL_DIALECT_DIRS "${BB_BALL_DIALECT_DIRS}")
include_directories(${BB_BALL_DIALECT_DIRS})

foreach(BB_BALL_SOURCE_KIND legalize assign tile)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
      --repo "${BB_REPO_ROOT}"
      --chip-pb "${BUCKYBALL_CHIP_PB}"
      --print-ball-compiler-paths "${BB_BALL_SOURCE_KIND}"
    RESULT_VARIABLE BB_BALL_SOURCE_RESULT
    OUTPUT_VARIABLE BB_BALL_SOURCE_PATHS
    ERROR_VARIABLE BB_BALL_SOURCE_ERROR)
  if(NOT BB_BALL_SOURCE_RESULT EQUAL 0)
    message(FATAL_ERROR
      "failed to resolve ${BB_BALL_SOURCE_KIND} Ball compiler sources:\n${BB_BALL_SOURCE_ERROR}")
  endif()
  string(STRIP "${BB_BALL_SOURCE_PATHS}" BB_BALL_SOURCE_PATHS)
  string(REPLACE "\n" ";" BB_${BB_BALL_SOURCE_KIND}_SOURCES
    "${BB_BALL_SOURCE_PATHS}")
endforeach()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-ball-compiler-paths bank-ssa
  RESULT_VARIABLE BB_BANK_SSA_RESULT
  OUTPUT_VARIABLE BB_BANK_SSA_SOURCES
  ERROR_VARIABLE BB_BANK_SSA_ERROR)
if(NOT BB_BANK_SSA_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to resolve bank-ssa Ball compiler sources:\n${BB_BANK_SSA_ERROR}")
endif()
string(STRIP "${BB_BANK_SSA_SOURCES}" BB_BANK_SSA_SOURCES)
string(REPLACE "\n" ";" BB_BANK_SSA_SOURCES "${BB_BANK_SSA_SOURCES}")

set(LLVM_TARGET_DEFINITIONS "${BB_TARGET_DEFINITIONS}")
mlir_tablegen(Buckyball.h.inc -gen-op-decls)
mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
mlir_tablegen(BuckyballTypes.h.inc -gen-typedef-decls -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
add_public_tablegen_target(MLIRBuckyballIncGen)
set(LLVM_TARGET_DEFINITIONS "${BB_TARGET_DEFINITIONS}")
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBuckyballConversionsIncGen)

add_mlir_dialect_library(BuddyBuckyball
  "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/IR/BuckyballDialect.cpp"
  src/Target/BuckyballTargetRegistry.cpp
  LINK_LIBS PUBLIC MLIRIR MLIRLLVMDialect)
target_include_directories(BuddyBuckyball PUBLIC
  "${BB_COMPILER_COMMON_DIR}"
  "${BB_COMPILER_COMMON_DIR}/Dialect"
  "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}")

add_mlir_dialect_library(BuddyBuckyballTransforms
  "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp"
  ${BB_legalize_SOURCES}
  src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  LINK_LIBS PUBLIC BuddyBuckyball MLIRIR MLIRLLVMCommonConversion MLIRLLVMDialect)

add_mlir_library(LowerBuckyballPass
  "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp"
  "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp"
  "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp"
  ${BB_assign_SOURCES}
  ${BB_BANK_SSA_SOURCES}
  src/Conversion/LowerBuckyball/AssignPhysicalBanksPass.cpp
  src/Conversion/LowerBuckyball/LowerBuckyballPass.cpp
  src/Conversion/LowerBuckyball/LowerBuckyballToBankSSAPass.cpp
  LINK_LIBS PUBLIC BuddyBuckyballTransforms MLIRLinalgDialect)

add_mlir_library(LowerTileToBuckyballPass
  src/Conversion/LowerTileToBuckyball/LowerTileToBuckyballPass.cpp
  ${BB_tile_SOURCES}
  LINK_LIBS PUBLIC BuddyTile BuddyBuckyball MLIRLinalgDialect)

add_mlir_translation_library(BuddyBuckyballToLLVMIRTranslation
  "${BB_COMPILER_COMMON_DIR}/Target/LLVMIR/Dialect/Buckyball/BuckyballToLLVMIRTranslation.cpp"
  DEPENDS BuddyBuckyballConversionsIncGen intrinsics_gen
  LINK_COMPONENTS Core
  LINK_LIBS PUBLIC MLIRIR MLIRSupport BuddyBuckyball)
