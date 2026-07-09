cmake_minimum_required(VERSION 3.12)
project(BuckyballTest C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)
set(BBSW_BAREMETAL_DIR ${WORKLOAD_LIB_DIR}/bbsw/baremetal)

set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})
set(BUDDY_MLIR_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/../)
set(BUDDY_BINARY_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)
set(LLVM_BUILD_DIR ${BUDDY_MLIR_DIR}/llvm/build)

get_filename_component(BUCKYBALL_REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)
set(BUCKYBALL_WORKLOAD_CHIP "" CACHE STRING "Buckyball chip for workloads")
if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
  message(FATAL_ERROR
    "BUCKYBALL_WORKLOAD_CHIP is required. Pass -DBUCKYBALL_WORKLOAD_CHIP=<chip>.")
endif()
if(NOT IS_DIRECTORY
    "${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}")
  message(FATAL_ERROR
    "Buckyball workload chip does not exist: ${BUCKYBALL_WORKLOAD_CHIP}")
endif()
message(STATUS "Buckyball workload chip: ${BUCKYBALL_WORKLOAD_CHIP}")

set(BUCKYBALL_EXPECTED_COMPILER_DIR
  "${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}/compiler")
set(BUCKYBALL_COMPILER_CACHE "$ENV{BUDDY_MLIR_BUILD_DIR}/CMakeCache.txt")
if(NOT EXISTS "${BUCKYBALL_COMPILER_CACHE}")
  message(FATAL_ERROR
    "Buddy MLIR CMakeCache.txt does not exist: ${BUCKYBALL_COMPILER_CACHE}")
endif()
file(STRINGS "${BUCKYBALL_COMPILER_CACHE}" BUCKYBALL_COMPILER_CACHE_LINES
  REGEX "^BUDDY_EXTERNAL_DIALECTS_DIR:PATH=")
list(LENGTH BUCKYBALL_COMPILER_CACHE_LINES BUCKYBALL_COMPILER_CACHE_COUNT)
if(NOT BUCKYBALL_COMPILER_CACHE_COUNT EQUAL 1)
  message(FATAL_ERROR
    "BUDDY_EXTERNAL_DIALECTS_DIR is missing or ambiguous in ${BUCKYBALL_COMPILER_CACHE}")
endif()
list(GET BUCKYBALL_COMPILER_CACHE_LINES 0 BUCKYBALL_COMPILER_CACHE_LINE)
string(REGEX REPLACE "^BUDDY_EXTERNAL_DIALECTS_DIR:PATH=" ""
  BUCKYBALL_COMPILER_DIR "${BUCKYBALL_COMPILER_CACHE_LINE}")
get_filename_component(BUCKYBALL_EXPECTED_COMPILER_DIR
  "${BUCKYBALL_EXPECTED_COMPILER_DIR}" ABSOLUTE)
get_filename_component(BUCKYBALL_COMPILER_DIR
  "${BUCKYBALL_COMPILER_DIR}" ABSOLUTE)
if(NOT BUCKYBALL_COMPILER_DIR STREQUAL BUCKYBALL_EXPECTED_COMPILER_DIR)
  message(FATAL_ERROR
    "Compiler chip mismatch: workload chip is ${BUCKYBALL_WORKLOAD_CHIP}, "
    "but BUDDY_EXTERNAL_DIALECTS_DIR is ${BUCKYBALL_COMPILER_DIR}. "
    "Run `bbdev compiler --build '--chip ${BUCKYBALL_WORKLOAD_CHIP}'` first.")
endif()

option(BUCKYBALL_STABLE "Use stable LLVM Buckyball extensions" OFF)
if(BUCKYBALL_STABLE)
  set(BUCKYBALL_RISCV_MATTR "+xbbpebble,+xbbmeteor,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL "--lower-buckyball=stable=true")
  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      "--lower-bank-ssa-to-intrinsics=stable=true")
else()
  set(BUCKYBALL_RISCV_MATTR "+xbuckyball,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL -lower-buckyball)
  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      -lower-bank-ssa-to-intrinsics)
endif()
message(STATUS "Buckyball RISC-V mattr: ${BUCKYBALL_RISCV_MATTR}")

set(ELF_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc)
set(LINUX_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)

set(BUILD_BIN_DIR ${WORKLOAD_DIR}/../build/workloads)
set(OUTPUT_BIN_DIR ${WORKLOAD_DIR}/../output/workloads)

set(Python3_EXECUTABLE python3)
#-------------------------------------------------------------------------------
# add_subdirectory
#-------------------------------------------------------------------------------
add_subdirectory(lib)
add_subdirectory(src)

#-------------------------------------------------------------------------------
# setup directory
#-------------------------------------------------------------------------------
file(MAKE_DIRECTORY ${OUTPUT_BIN_DIR})

#-------------------------------------------------------------------------------
# set custom target
#-------------------------------------------------------------------------------
add_custom_target(create-dirs
  COMMAND mkdir -p ${OUTPUT_BIN_DIR}
  COMMENT "Creating necessary directories"
)

add_custom_target(sync-bin
  COMMAND rsync -av --delete --delete-excluded
          --exclude="CMakeFiles"
          --include="**/chips/${BUCKYBALL_WORKLOAD_CHIP}/"
          --exclude="**/chips/*/"
          --include="*/"
          --include="*-baremetal"
          --include="*-linux"
          --include="*-run"
          --exclude="*"
          ${BUILD_BIN_DIR}/ ${OUTPUT_BIN_DIR}/
  COMMENT "Syncing binary files to output directory"
)

add_dependencies(sync-bin
  base-ctests-build
  tutorial-build
  chip-workloads-build
  OpTest-all
  embench-all
  coremark-all
)

add_custom_target(build-all ALL
  DEPENDS sync-bin
)

add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND rm -rf ${OUTPUT_BIN_DIR}
  COMMENT "Cleaning all build artifacts"
)
