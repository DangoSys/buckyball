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
set(BUCKYBALL_WORKLOAD_CORE "" CACHE STRING "Compiler Core for workloads")
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
if(BUCKYBALL_WORKLOAD_CORE STREQUAL "")
  message(FATAL_ERROR
    "BUCKYBALL_WORKLOAD_CORE is required. bbdev derives it from chip.toml.")
endif()
message(STATUS "Buckyball workload compiler Core: ${BUCKYBALL_WORKLOAD_CORE}")

set(BUCKYBALL_CORE_DIR
  "${BUCKYBALL_REPO_ROOT}/examples/cores/${BUCKYBALL_WORKLOAD_CORE}")
set(BUCKYBALL_CORE_MANIFEST "${BUCKYBALL_CORE_DIR}/manifest.toml")
if(NOT EXISTS "${BUCKYBALL_CORE_MANIFEST}")
  message(FATAL_ERROR "Core manifest does not exist: ${BUCKYBALL_CORE_MANIFEST}")
endif()

# Compiler hardware parameters come from the Core memory-domain TOML. Keep
# this small parser here because the workload configure step already owns the
# selected Core and CMake has no built-in TOML reader.
file(STRINGS "${BUCKYBALL_CORE_MANIFEST}" BUCKYBALL_CORE_CONFIG_LINE
  REGEX "^[ \t]*config[ \t]*=")
list(LENGTH BUCKYBALL_CORE_CONFIG_LINE BUCKYBALL_CORE_CONFIG_COUNT)
if(NOT BUCKYBALL_CORE_CONFIG_COUNT EQUAL 1)
  message(FATAL_ERROR
    "${BUCKYBALL_CORE_MANIFEST} must define exactly one core config entry")
endif()
list(GET BUCKYBALL_CORE_CONFIG_LINE 0 BUCKYBALL_CORE_CONFIG_LINE)
string(REGEX REPLACE "^[^=]+=[ \t]*\"([^\"]+)\".*" "\\1"
  BUCKYBALL_CORE_CONFIG_REL "${BUCKYBALL_CORE_CONFIG_LINE}")
set(BUCKYBALL_CORE_CONFIG_TOML
  "${BUCKYBALL_CORE_DIR}/${BUCKYBALL_CORE_CONFIG_REL}")
if(NOT EXISTS "${BUCKYBALL_CORE_CONFIG_TOML}")
  message(FATAL_ERROR "Core config TOML does not exist: ${BUCKYBALL_CORE_CONFIG_TOML}")
endif()
get_filename_component(BUCKYBALL_CORE_CONFIG_DIR
  "${BUCKYBALL_CORE_CONFIG_TOML}" DIRECTORY)

file(STRINGS "${BUCKYBALL_CORE_CONFIG_TOML}" BUCKYBALL_MEMDOMAIN_LINE
  REGEX "^[ \t]*memdomain[ \t]*=")
list(LENGTH BUCKYBALL_MEMDOMAIN_LINE BUCKYBALL_MEMDOMAIN_COUNT)
if(NOT BUCKYBALL_MEMDOMAIN_COUNT EQUAL 1)
  message(FATAL_ERROR
    "${BUCKYBALL_CORE_CONFIG_TOML} must define exactly one memdomain entry")
endif()
list(GET BUCKYBALL_MEMDOMAIN_LINE 0 BUCKYBALL_MEMDOMAIN_LINE)
string(REGEX REPLACE "^[^=]+=[ \t]*\"([^\"]+)\".*" "\\1"
  BUCKYBALL_MEMDOMAIN_REL "${BUCKYBALL_MEMDOMAIN_LINE}")
set(BUCKYBALL_MEMDOMAIN_TOML
  "${BUCKYBALL_CORE_CONFIG_DIR}/${BUCKYBALL_MEMDOMAIN_REL}")
if(NOT EXISTS "${BUCKYBALL_MEMDOMAIN_TOML}")
  message(FATAL_ERROR "Memory-domain TOML does not exist: ${BUCKYBALL_MEMDOMAIN_TOML}")
endif()

file(STRINGS "${BUCKYBALL_MEMDOMAIN_TOML}" BUCKYBALL_MEMDOMAIN_LINES)
set(BUCKYBALL_IN_BANK_SECTION FALSE)
foreach(BUCKYBALL_MEMDOMAIN_LINE IN LISTS BUCKYBALL_MEMDOMAIN_LINES)
  # Strip comments and surrounding whitespace before interpreting the TOML.
  string(REGEX REPLACE "#.*$" "" BUCKYBALL_MEMDOMAIN_LINE
    "${BUCKYBALL_MEMDOMAIN_LINE}")
  string(STRIP "${BUCKYBALL_MEMDOMAIN_LINE}" BUCKYBALL_MEMDOMAIN_LINE)
  if(BUCKYBALL_MEMDOMAIN_LINE STREQUAL "[bank]")
    set(BUCKYBALL_IN_BANK_SECTION TRUE)
    continue()
  endif()
  if(BUCKYBALL_MEMDOMAIN_LINE MATCHES "^\\[")
    set(BUCKYBALL_IN_BANK_SECTION FALSE)
  endif()
  if(BUCKYBALL_IN_BANK_SECTION AND
     BUCKYBALL_MEMDOMAIN_LINE MATCHES "^(num|width|entries)[ \t]*=[ \t]*[0-9]+$")
    string(REGEX MATCH "^([^=]+)[ \t]*=[ \t]*([0-9]+)$"
      BUCKYBALL_BANK_MATCH "${BUCKYBALL_MEMDOMAIN_LINE}")
    string(STRIP "${CMAKE_MATCH_1}" BUCKYBALL_BANK_KEY)
    set(BUCKYBALL_BANK_VALUE "${CMAKE_MATCH_2}")
    if(BUCKYBALL_BANK_KEY STREQUAL "num")
      set(BUCKYBALL_MLIR_BANK_NUM "${BUCKYBALL_BANK_VALUE}")
    elseif(BUCKYBALL_BANK_KEY STREQUAL "width")
      set(BUCKYBALL_MLIR_BANK_WIDTH_BITS "${BUCKYBALL_BANK_VALUE}")
    elseif(BUCKYBALL_BANK_KEY STREQUAL "entries")
      set(BUCKYBALL_MLIR_BANK_DEPTH "${BUCKYBALL_BANK_VALUE}")
    endif()
  endif()
endforeach()
foreach(BUCKYBALL_BANK_PARAMETER IN ITEMS
    BUCKYBALL_MLIR_BANK_NUM BUCKYBALL_MLIR_BANK_WIDTH_BITS BUCKYBALL_MLIR_BANK_DEPTH)
  if(NOT DEFINED ${BUCKYBALL_BANK_PARAMETER})
    message(FATAL_ERROR
      "${BUCKYBALL_MEMDOMAIN_TOML} [bank] is missing ${BUCKYBALL_BANK_PARAMETER}")
  endif()
endforeach()
math(EXPR BUCKYBALL_MLIR_BANK_WIDTH_BYTES
  "${BUCKYBALL_MLIR_BANK_WIDTH_BITS} / 8")
math(EXPR BUCKYBALL_MLIR_BANK_WIDTH_REMAINDER
  "${BUCKYBALL_MLIR_BANK_WIDTH_BITS} % 8")
if(BUCKYBALL_MLIR_BANK_WIDTH_BITS LESS 8 OR
   NOT BUCKYBALL_MLIR_BANK_WIDTH_REMAINDER EQUAL 0)
  message(FATAL_ERROR
    "Core bank width must be a positive whole number of bytes: ${BUCKYBALL_MLIR_BANK_WIDTH_BITS}")
endif()
set(BUCKYBALL_CONVERT_TILE_TO_BUCKYBALL
  "--convert-tile-to-buckyball=bank_width=${BUCKYBALL_MLIR_BANK_WIDTH_BYTES} bank_depth=${BUCKYBALL_MLIR_BANK_DEPTH} bank_num=${BUCKYBALL_MLIR_BANK_NUM}")
set(BUCKYBALL_ASSIGN_PHYSICAL_BANKS
  "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}")
message(STATUS "Buckyball Core banks: ${BUCKYBALL_MLIR_BANK_NUM} x ${BUCKYBALL_MLIR_BANK_WIDTH_BITS}-bit x ${BUCKYBALL_MLIR_BANK_DEPTH}")

set(BUCKYBALL_RUSHB_CHIP_CMAKE
    ${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}/workloads/cmake/RushB.cmake)
if(EXISTS ${BUCKYBALL_RUSHB_CHIP_CMAKE})
  include(${BUCKYBALL_RUSHB_CHIP_CMAKE})
endif()

if(DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST)
  add_custom_target(rushB-bemu-workloads-build)
endif()
if(DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
  add_custom_target(rushB-verilator-workloads-build)
endif()

set(BUCKYBALL_EXPECTED_COMPILER_DIR
  "${BUCKYBALL_REPO_ROOT}/examples/cores/${BUCKYBALL_WORKLOAD_CORE}/compiler")
set(BUCKYBALL_COMPILER_CACHE "$ENV{BUDDY_MLIR_BUILD_DIR}/CMakeCache.txt")
if(NOT EXISTS "${BUCKYBALL_COMPILER_CACHE}")
  message(FATAL_ERROR
    "Buddy MLIR CMakeCache.txt does not exist: ${BUCKYBALL_COMPILER_CACHE}")
endif()
file(STRINGS "${BUCKYBALL_COMPILER_CACHE}" BUCKYBALL_COMPILER_CACHE_LINES
  REGEX "^BUDDY_EXTERNAL_DIALECTS_DIR:[^=]+=")
list(LENGTH BUCKYBALL_COMPILER_CACHE_LINES BUCKYBALL_COMPILER_CACHE_COUNT)
if(NOT BUCKYBALL_COMPILER_CACHE_COUNT EQUAL 1)
  message(FATAL_ERROR
    "BUDDY_EXTERNAL_DIALECTS_DIR is missing or ambiguous in ${BUCKYBALL_COMPILER_CACHE}")
endif()
list(GET BUCKYBALL_COMPILER_CACHE_LINES 0 BUCKYBALL_COMPILER_CACHE_LINE)
string(REGEX REPLACE "^BUDDY_EXTERNAL_DIALECTS_DIR:[^=]+=" ""
  BUCKYBALL_COMPILER_DIR "${BUCKYBALL_COMPILER_CACHE_LINE}")
get_filename_component(BUCKYBALL_EXPECTED_COMPILER_DIR
  "${BUCKYBALL_EXPECTED_COMPILER_DIR}" ABSOLUTE)
get_filename_component(BUCKYBALL_COMPILER_DIR
  "${BUCKYBALL_COMPILER_DIR}" ABSOLUTE)
if(NOT BUCKYBALL_COMPILER_DIR STREQUAL BUCKYBALL_EXPECTED_COMPILER_DIR)
  message(FATAL_ERROR
    "Compiler Core mismatch: workload chip is ${BUCKYBALL_WORKLOAD_CHIP}, "
    "whose compiler Core is ${BUCKYBALL_WORKLOAD_CORE}, "
    "but BUDDY_EXTERNAL_DIALECTS_DIR is ${BUCKYBALL_COMPILER_DIR}. "
    "Run `bbdev compiler --build '--core ${BUCKYBALL_WORKLOAD_CORE}'` first.")
endif()

option(BUCKYBALL_STABLE "Use stable LLVM Buckyball extensions" OFF)
if(BUCKYBALL_STABLE)
  set(BUCKYBALL_RISCV_MATTR "+xbbpebble,+xbbmeteor,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL "--lower-buckyball=stable=true")
  set(BUCKYBALL_LOWER_BUCKYBALL_RUSHB
      "--lower-buckyball=stable=true rushb=true")
  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      "--lower-bank-ssa-to-intrinsics=stable=true")
  set(BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS
      "--lower-bank-ssa-to-intrinsics=stable=true rushb=true")
else()
  set(BUCKYBALL_RISCV_MATTR "+xbuckyball,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL -lower-buckyball)
  set(BUCKYBALL_LOWER_BUCKYBALL_RUSHB "--lower-buckyball=rushb=true")
  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      -lower-bank-ssa-to-intrinsics)
  set(BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS
      "--lower-bank-ssa-to-intrinsics=rushb=true")
endif()
message(STATUS "Buckyball RISC-V mattr: ${BUCKYBALL_RISCV_MATTR}")

set(ELF_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc)
set(LINUX_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)

set(BUILD_BIN_DIR ${WORKLOAD_DIR}/../build/workloads)
set(OUTPUT_BIN_DIR ${WORKLOAD_DIR}/../output/workloads)

if(NOT IS_ABSOLUTE "${Python3_EXECUTABLE}")
  find_program(BUCKYBALL_PYTHON3_EXECUTABLE NAMES python3)
  if(NOT BUCKYBALL_PYTHON3_EXECUTABLE)
    message(FATAL_ERROR "python3 is required to build workloads")
  endif()
  set(Python3_EXECUTABLE "${BUCKYBALL_PYTHON3_EXECUTABLE}")
endif()
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
          --filter="P_src/CTest/rushB/"
          --filter="P_src/CTest/rushB/***"
          --filter="P_src/OpTest/rushB/"
          --filter="P_src/OpTest/rushB/***"
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
