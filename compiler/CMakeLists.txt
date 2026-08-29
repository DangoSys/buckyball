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
set(BB_BUCKYBALL_DIALECT_TD "${CMAKE_CURRENT_BINARY_DIR}/BuckyballUnion.td")
set(BB_BALL_LOWERING_HOOKS
  "${CMAKE_CURRENT_BINARY_DIR}/BuckyballBallLoweringHooks.inc")
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
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --lowering-hooks-out "${BB_BALL_LOWERING_HOOKS}"
  RESULT_VARIABLE BB_BALL_HOOKS_RESULT
  OUTPUT_VARIABLE BB_BALL_HOOKS_OUTPUT
  ERROR_VARIABLE BB_BALL_HOOKS_ERROR)
if(NOT BB_BALL_HOOKS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to generate Buckyball lowering hooks:\n${BB_BALL_HOOKS_OUTPUT}${BB_BALL_HOOKS_ERROR}")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --out-dialect-td "${BB_BUCKYBALL_DIALECT_TD}"
  RESULT_VARIABLE BB_DIALECT_TD_RESULT
  OUTPUT_VARIABLE BB_DIALECT_TD_OUTPUT
  ERROR_VARIABLE BB_DIALECT_TD_ERROR)
if(NOT BB_DIALECT_TD_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to generate Buckyball union dialect:\n${BB_DIALECT_TD_OUTPUT}${BB_DIALECT_TD_ERROR}")
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
    --print-ball-dialect-dirs
  RESULT_VARIABLE BB_BALL_DIALECT_DIRS_RESULT
  OUTPUT_VARIABLE BB_BALL_DIALECT_DIRS
  ERROR_VARIABLE BB_BALL_DIALECT_DIRS_ERROR)
if(NOT BB_BALL_DIALECT_DIRS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to resolve Buckyball Ball dialect directories:\n${BB_BALL_DIALECT_DIRS_ERROR}")
endif()
string(STRIP "${BB_BALL_DIALECT_DIRS}" BB_BALL_DIALECT_DIRS)
string(REPLACE "\n" ";" BB_BALL_DIALECT_DIRS "${BB_BALL_DIALECT_DIRS}")
foreach(BB_BALL_DIALECT_DIR ${BB_BALL_DIALECT_DIRS})
  include_directories("${BB_BALL_DIALECT_DIR}")
endforeach()

# Both single-core and heterogeneous builds use the generated hook names in
# AssignPhysicalBanksPass.  Compile only the Ball assign patterns selected by
# this Chip.pb; the framework itself keeps no Ball list.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${BB_TARGET_REGISTRY_GENERATOR}"
    --repo "${BB_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-ball-compiler-paths assign
  RESULT_VARIABLE BB_ASSIGN_SOURCES_RESULT
  OUTPUT_VARIABLE BB_ASSIGN_SOURCES_RAW
  ERROR_VARIABLE BB_ASSIGN_SOURCES_ERROR)
if(NOT BB_ASSIGN_SOURCES_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to resolve assign Ball compiler sources:\n${BB_ASSIGN_SOURCES_ERROR}")
endif()
string(STRIP "${BB_ASSIGN_SOURCES_RAW}" BB_ASSIGN_SOURCES_RAW)
string(REPLACE "\n" ";" BB_assign_SOURCES "${BB_ASSIGN_SOURCES_RAW}")

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
if(BB_CORE_COMPILER_DIR_COUNT EQUAL 1)
  list(GET BB_CORE_COMPILER_DIRS 0 BB_CORE_COMPILER_DIR)
  get_filename_component(BB_CORE_COMPILER_NAME "${BB_CORE_COMPILER_DIR}" NAME)
  add_subdirectory("${BB_CORE_COMPILER_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/cores/${BB_CORE_COMPILER_NAME}")
else()
  # A heterogeneous chip owns more than one Core lowering policy.  Build its
  # dialect and public passes once; the selected target chooses its Ball set
  # from Chip.pb at pass execution time.
  foreach(BB_BALL_SOURCE_KIND legalize bank_ssa tile)
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

  set(LLVM_TARGET_DEFINITIONS "${BB_BUCKYBALL_DIALECT_TD}")
  mlir_tablegen(Buckyball.h.inc -gen-op-decls)
  mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
  mlir_tablegen(BuckyballTypes.h.inc -gen-typedef-decls -typedefs-dialect=buckyball)
  mlir_tablegen(BuckyballTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=buckyball)
  mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
  mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
  add_public_tablegen_target(MLIRBuckyballIncGen)
  mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
  add_public_tablegen_target(BuddyBuckyballConversionsIncGen)

  add_mlir_dialect_library(BuddyBuckyball
    "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/IR/BuckyballDialect.cpp"
    "${BB_TARGET_REGISTRY_SOURCE}"
    LINK_LIBS PUBLIC MLIRIR MLIRLLVMDialect)
  target_include_directories(BuddyBuckyball PUBLIC
    "${BB_COMPILER_COMMON_DIR}"
    "${BB_COMPILER_COMMON_DIR}/Dialect"
    "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball"
    "${CMAKE_CURRENT_BINARY_DIR}")

  add_mlir_dialect_library(BuddyBuckyballTransforms
    "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp"
    ${BB_legalize_SOURCES}
    "${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp"
    LINK_LIBS PUBLIC BuddyBuckyball MLIRIR MLIRLLVMCommonConversion MLIRLLVMDialect)

  add_mlir_library(LowerBuckyballPass
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp"
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp"
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp"
    ${BB_assign_SOURCES}
    ${BB_bank_ssa_SOURCES}
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/AssignPhysicalBanksPass.cpp"
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/LowerBuckyballPass.cpp"
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/LowerBuckyballToBankSSAPass.cpp"
    LINK_LIBS PUBLIC BuddyBuckyballTransforms MLIRLinalgDialect)

  add_mlir_library(LowerTileToBuckyballPass
    "${BB_COMPILER_COMMON_DIR}/Conversion/LowerTileToBuckyball/LowerTileToBuckyballPass.cpp"
    ${BB_tile_SOURCES}
    LINK_LIBS PUBLIC BuddyTile BuddyBuckyball MLIRLinalgDialect)

  add_mlir_translation_library(BuddyBuckyballToLLVMIRTranslation
    "${BB_COMPILER_COMMON_DIR}/Target/LLVMIR/Dialect/Buckyball/BuckyballToLLVMIRTranslation.cpp"
    DEPENDS BuddyBuckyballConversionsIncGen intrinsics_gen
    LINK_COMPONENTS Core
    LINK_LIBS PUBLIC MLIRIR MLIRSupport BuddyBuckyball)
endif()
