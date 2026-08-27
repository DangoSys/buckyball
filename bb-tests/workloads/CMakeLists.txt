cmake_minimum_required(VERSION 3.12)
project(BuckyballTest C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)

get_filename_component(BUCKYBALL_REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)
set(BBSW_BAREMETAL_DIR ${WORKLOAD_LIB_DIR}/bbsw/baremetal)

set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})
set(BUDDY_MLIR_DIR ${BUCKYBALL_REPO_ROOT}/compiler/thirdparty/buddy-mlir)
set(BUDDY_BINARY_DIR $ENV{BUDDY_MLIR_BUILD_DIR}/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)
set(LLVM_BUILD_DIR ${BUDDY_MLIR_DIR}/llvm/build)

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

if(NOT IS_ABSOLUTE "${Python3_EXECUTABLE}")
  find_program(BUCKYBALL_PYTHON3_EXECUTABLE NAMES python3)
  if(NOT BUCKYBALL_PYTHON3_EXECUTABLE)
    message(FATAL_ERROR "python3 is required to build workloads")
  endif()
  set(Python3_EXECUTABLE "${BUCKYBALL_PYTHON3_EXECUTABLE}")
endif()

foreach(_BUCKYBALL_REQ IN ITEMS
    BUCKYBALL_CHIP_PB)
  if(NOT DEFINED ${_BUCKYBALL_REQ} OR "${${_BUCKYBALL_REQ}}" STREQUAL "")
    message(FATAL_ERROR
      "${_BUCKYBALL_REQ} must be passed by bbdev config --install (cmake.defs)")
  endif()
endforeach()

if(NOT EXISTS "${BUCKYBALL_CHIP_PB}")
  message(FATAL_ERROR "missing Chip.pb: ${BUCKYBALL_CHIP_PB}")
endif()
set(BUCKYBALL_ISA_DIR "${CMAKE_BINARY_DIR}/generated/isa")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${BUCKYBALL_REPO_ROOT}/compiler/scripts/pb_to_target_registry.py"
    --repo "${BUCKYBALL_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --isa-dir "${BUCKYBALL_ISA_DIR}"
    --print-targets
    --print-target-balls
    --print-core-targets
  RESULT_VARIABLE BUCKYBALL_TARGET_REGISTRY_RESULT
  OUTPUT_VARIABLE BUCKYBALL_TARGETS_RAW
  ERROR_VARIABLE BUCKYBALL_TARGET_REGISTRY_ERROR)
if(NOT BUCKYBALL_TARGET_REGISTRY_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to generate target ISA headers: ${BUCKYBALL_TARGET_REGISTRY_ERROR}")
endif()
string(REPLACE "\n" ";" BUCKYBALL_TARGETS "${BUCKYBALL_TARGETS_RAW}")
set(BUCKYBALL_TARGET_BALL_LINES)
set(BUCKYBALL_CORE_TARGET_LINES)
foreach(BUCKYBALL_TARGET_LINE ${BUCKYBALL_TARGETS})
  if(BUCKYBALL_TARGET_LINE MATCHES "^([0-9]+):([^,]+)$")
    set(BUCKYBALL_CORE_TARGET_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    list(APPEND BUCKYBALL_CORE_TARGET_LINES ${BUCKYBALL_TARGET_LINE})
  elseif(BUCKYBALL_TARGET_LINE MATCHES "^([^:]+):(.*)$")
    set(BUCKYBALL_TARGET_BALLS_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    list(APPEND BUCKYBALL_TARGET_BALL_LINES ${BUCKYBALL_TARGET_LINE})
  endif()
endforeach()
list(REMOVE_ITEM BUCKYBALL_TARGETS ${BUCKYBALL_TARGET_BALL_LINES}
  ${BUCKYBALL_CORE_TARGET_LINES} "")
if(NOT BUCKYBALL_TARGETS)
  message(FATAL_ERROR "Chip.pb has no compiler targets")
endif()
message(STATUS "Buckyball compiler targets: ${BUCKYBALL_TARGETS}")
set(BUCKYBALL_BALL_NAMES
  fp2int:Fp2IntBall
  im2col:Im2colBall
  int2fp:Int2FpBall
  matadd:MatAddBall
  relu:ReluBall
  smatmul:SMatMulBall
  transpose:TransposeBall
  vector:VecBall)
set(BUCKYBALL_MLIR_BALL_NAMES
  fp2int:Fp2IntBall
  im2col:Im2colBall
  int2fp:Int2FpBall
  matadd:MatAddBall
  relu:ReluBall
  smatmul:SMatMulBall
  transpose:TransposeBall)

add_custom_target(rushB-bemu-workloads-build)
add_custom_target(rushB-verilator-workloads-build)

if(NOT DEFINED ELF_CC)
  set(ELF_CC "riscv64-unknown-elf-gcc")
endif()

if(NOT DEFINED LINUX_CC)
  set(LINUX_CC "riscv64-unknown-linux-gnu-gcc")
endif()

set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(BUDDY_CYCLE_TRACE_RUNTIME_SRC ${BBSW_BAREMETAL_DIR}/buddy_cycle_trace.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker")

function(add_buckyball_rushb_native TARGET_NAME)
  cmake_parse_arguments(ARG "CXX" "OUTPUT_SUBDIR" "SOURCES;INCLUDE_DIRS;DEPENDS" ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "${TARGET_NAME}: rushB runner has no sources")
  endif()
  if(NOT ARG_OUTPUT_SUBDIR)
    message(FATAL_ERROR "${TARGET_NAME}: rushB runner has no output directory")
  endif()

  if(NOT DEFINED BUILD_BIN_DIR OR NOT DEFINED OUTPUT_BIN_DIR)
    message(FATAL_ERROR
      "add_buckyball_rushb_native requires BUILD_BIN_DIR and OUTPUT_BIN_DIR")
  endif()

  set(RUSHB_OUTPUT_ROOT ${OUTPUT_BIN_DIR}/${ARG_OUTPUT_SUBDIR})
  set(RUSHB_SYNC_ROOT ${BUILD_BIN_DIR}/${ARG_OUTPUT_SUBDIR})
  string(MAKE_C_IDENTIFIER "${ARG_OUTPUT_SUBDIR}" RUSHB_OUTPUT_ROOT_ID)
  set(RUSHB_OUTPUT_ROOT_TARGET rushB-output-${RUSHB_OUTPUT_ROOT_ID})
  if(NOT TARGET ${RUSHB_OUTPUT_ROOT_TARGET})
    add_custom_target(${RUSHB_OUTPUT_ROOT_TARGET}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_OUTPUT_ROOT}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_SYNC_ROOT}
      COMMENT "Creating rushB output directory: ${ARG_OUTPUT_SUBDIR}"
      VERBATIM)
  endif()

  if(ARG_CXX)
    set(RUSHB_COMPILER c++)
    set(RUSHB_STANDARD -std=c++17)
  else()
    set(RUSHB_COMPILER cc)
    set(RUSHB_STANDARD -std=gnu17)
  endif()

  foreach(RUSHB_BACKEND bemu verilator)
    set(RUSHB_TARGET ${TARGET_NAME}-rushB-${RUSHB_BACKEND}-run)
    set(RUSHB_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/rushB/${RUSHB_TARGET})
    set(RUSHB_BINARY ${RUSHB_BUILD_DIR}/${RUSHB_TARGET}.bin)
    set(RUSHB_OUTPUT_DIR ${RUSHB_OUTPUT_ROOT}/${RUSHB_TARGET})
    set(RUSHB_SYNC_DIR ${RUSHB_SYNC_ROOT}/${RUSHB_TARGET})
    if(RUSHB_BACKEND STREQUAL "bemu")
      set(RUSHB_LIBRARY ${BUCKYBALL_RUSHB_BEMU_LIBRARY})
      set(RUSHB_LIBRARY_NAME bebop_bemu)
      set(RUSHB_BUILD_RUNTIME
        COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${BUCKYBALL_CARGO_TARGET_DIR}
                cargo build --release --manifest-path ${BUCKYBALL_RUSHB_BEMU_MANIFEST} --lib)
      set(RUSHB_RUNTIME_DEPENDENCY ${BUCKYBALL_RUSHB_BEMU_MANIFEST})
    else()
      set(RUSHB_LIBRARY ${BUCKYBALL_RUSHB_VERILATOR_LIBRARY})
      set(RUSHB_LIBRARY_NAME bebop_verilator)
      set(RUSHB_BUILD_RUNTIME)
      set(RUSHB_RUNTIME_DEPENDENCY ${BUCKYBALL_RUSHB_VERILATOR_LIBRARY})
    endif()
    if(NOT RUSHB_LIBRARY)
      message(FATAL_ERROR
        "${TARGET_NAME}: BUCKYBALL_RUSHB_${RUSHB_BACKEND} library is unset. "
        "RushB paths must be passed by Bazel/bbdev (-DBUCKYBALL_RUSHB_*).")
    endif()

    set(RUSHB_RUNTIME_OUTPUT
      ${RUSHB_OUTPUT_ROOT}/lib${RUSHB_LIBRARY_NAME}.so)
    set(RUSHB_RUNTIME_SYNC
      ${RUSHB_SYNC_ROOT}/lib${RUSHB_LIBRARY_NAME}.so)
    set(RUSHB_RUNTIME_TARGET
      rushB-runtime-${RUSHB_OUTPUT_ROOT_ID}-${RUSHB_BACKEND})
    if(NOT TARGET ${RUSHB_RUNTIME_TARGET})
      add_custom_command(
        OUTPUT ${RUSHB_RUNTIME_OUTPUT} ${RUSHB_RUNTIME_SYNC}
        ${RUSHB_BUILD_RUNTIME}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_OUTPUT_ROOT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_SYNC_ROOT}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${RUSHB_LIBRARY} ${RUSHB_RUNTIME_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${RUSHB_LIBRARY} ${RUSHB_RUNTIME_SYNC}
        DEPENDS ${RUSHB_RUNTIME_DEPENDENCY}
        COMMENT "Installing rushB ${RUSHB_BACKEND} runtime: ${ARG_OUTPUT_SUBDIR}"
        VERBATIM)
      add_custom_target(${RUSHB_RUNTIME_TARGET}
        DEPENDS ${RUSHB_RUNTIME_OUTPUT} ${RUSHB_RUNTIME_SYNC})
      add_dependencies(${RUSHB_RUNTIME_TARGET} ${RUSHB_OUTPUT_ROOT_TARGET})
    endif()

    get_filename_component(RUSHB_LIBRARY_DIR ${RUSHB_LIBRARY} DIRECTORY)
    set(RUSHB_INCLUDE_ARGS -I${BUCKYBALL_REPO_ROOT}/compiler/include)
    foreach(RUSHB_INCLUDE_DIR ${ARG_INCLUDE_DIRS})
      list(APPEND RUSHB_INCLUDE_ARGS -I${RUSHB_INCLUDE_DIR})
    endforeach()
    set(RUSHB_INSTALLED ${RUSHB_OUTPUT_DIR}/${RUSHB_TARGET})
    set(RUSHB_SYNCED ${RUSHB_SYNC_DIR}/${RUSHB_TARGET})
    add_custom_command(
      OUTPUT ${RUSHB_BINARY} ${RUSHB_INSTALLED} ${RUSHB_SYNCED}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_BUILD_DIR}
      COMMAND ${RUSHB_COMPILER} -no-pie ${RUSHB_STANDARD} -O2 -DBUCKYBALL_RUSHB
              ${RUSHB_INCLUDE_ARGS}
              ${ARG_SOURCES} ${BUCKYBALL_REPO_ROOT}/compiler/lib/RushBRuntime.c
              -L${RUSHB_LIBRARY_DIR} -l${RUSHB_LIBRARY_NAME}
              -Wl,-rpath,${RUSHB_OUTPUT_ROOT} -o ${RUSHB_BINARY}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_OUTPUT_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_SYNC_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUSHB_BINARY}
              ${RUSHB_INSTALLED}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUSHB_BINARY}
              ${RUSHB_SYNCED}
      DEPENDS ${ARG_SOURCES} ${ARG_DEPENDS}
              ${BUCKYBALL_REPO_ROOT}/compiler/include/buckyball/rushb.h
              ${BUCKYBALL_REPO_ROOT}/compiler/lib/RushBRuntime.c
      COMMENT "Building rushB ${RUSHB_BACKEND}: ${TARGET_NAME}"
      VERBATIM)
    add_custom_target(${RUSHB_TARGET}
      DEPENDS ${RUSHB_BINARY} ${RUSHB_INSTALLED} ${RUSHB_SYNCED})
    add_dependencies(${RUSHB_TARGET} ${RUSHB_OUTPUT_ROOT_TARGET})
    add_dependencies(${RUSHB_TARGET} ${RUSHB_RUNTIME_TARGET})
    add_dependencies(rushB-${RUSHB_BACKEND}-workloads-build ${RUSHB_TARGET})
  endforeach()
endfunction()

function(buckyball_add_ball_ctest_subdirs)
  foreach(BUCKYBALL_TARGET IN LISTS BUCKYBALL_TARGETS)
    foreach(BUCKYBALL_BALL_MAPPING IN LISTS BUCKYBALL_BALL_NAMES)
      string(REPLACE ":" ";" BUCKYBALL_BALL_PARTS "${BUCKYBALL_BALL_MAPPING}")
      list(GET BUCKYBALL_BALL_PARTS 0 BUCKYBALL_BALL_DIR)
      list(GET BUCKYBALL_BALL_PARTS 1 BUCKYBALL_BALL_CLASS)
      string(REPLACE "," ";" BUCKYBALL_ACTIVE_BALLS
        "${BUCKYBALL_TARGET_BALLS_${BUCKYBALL_TARGET}}")
      list(FIND BUCKYBALL_ACTIVE_BALLS "${BUCKYBALL_BALL_CLASS}"
        BUCKYBALL_BALL_INDEX)
      if(BUCKYBALL_BALL_INDEX EQUAL -1)
        continue()
      endif()
      set(BUCKYBALL_BALL_CTEST_DIR
        ${BUCKYBALL_REPO_ROOT}/examples/balls/${BUCKYBALL_BALL_DIR}/workloads/ctests)
      if(NOT IS_DIRECTORY ${BUCKYBALL_BALL_CTEST_DIR})
        message(FATAL_ERROR "missing Ball CTest directory: ${BUCKYBALL_BALL_CTEST_DIR}")
      endif()
      set(BUCKYBALL_CTEST_TARGET ${BUCKYBALL_TARGET})
      add_subdirectory(
        ${BUCKYBALL_BALL_CTEST_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_TARGET}/${BUCKYBALL_BALL_DIR}/ctests)
    endforeach()
  endforeach()
endfunction()

function(add_toy_mlir_rushb)
  cmake_parse_arguments(ARG "CXX" "NAME;TARGET_STEM;TARGET"
    "FRONT_PASSES;SOURCES;INCLUDE_DIRS;DEPENDS" ${ARGN})

  if(NOT ARG_NAME OR NOT ARG_TARGET_STEM OR NOT ARG_TARGET)
    message(FATAL_ERROR "add_toy_mlir_rushb requires NAME, TARGET_STEM, and TARGET")
  endif()
  list(FIND BUCKYBALL_TARGETS "${ARG_TARGET}" _buckyball_target_index)
  if(_buckyball_target_index EQUAL -1)
    message(FATAL_ERROR "unknown Buckyball target: ${ARG_TARGET}")
  endif()
  if(NOT DEFINED BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS OR
     NOT DEFINED BUCKYBALL_LOWER_BUCKYBALL_RUSHB)
    message(FATAL_ERROR
      "add_toy_mlir_rushb requires BUCKYBALL_LOWER_*_RUSHB")
  endif()
  if(NOT DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST OR
     NOT DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
    return()
  endif()

  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_NAME}.mlir)
  set(RUSHB_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}-rushB.o)

  add_custom_command(
    OUTPUT ${RUSHB_OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      --target=${ARG_TARGET}
      ${ARG_FRONT_PASSES}
      -assign-physical-banks
      ${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS}
      -convert-linalg-to-loops
      -expand-strided-metadata
      -lower-affine
      -convert-scf-to-cf
      -convert-cf-to-llvm
      ${BUCKYBALL_LOWER_BUCKYBALL_RUSHB}
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -lower-buckyball-intrinsics-to-rushb
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=x86_64 -O2 -o ${RUSHB_OBJ}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT} ${BUDDY_TRANSLATE} ${BUDDY_LLC}
            ${ARG_DEPENDS}
    COMMENT "Building rushB ${ARG_NAME}.o from Toy MLIR"
    VERBATIM)

  set(RUSHB_CXX_FLAG)
  if(ARG_CXX)
    set(RUSHB_CXX_FLAG CXX)
  endif()

  add_buckyball_rushb_native(${ARG_TARGET_STEM}
    ${RUSHB_CXX_FLAG}
    OUTPUT_SUBDIR src/CTest/rushB
    SOURCES ${ARG_SOURCES} ${RUSHB_OBJ}
    INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
    DEPENDS ${RUSHB_OBJ} ${ARG_DEPENDS})
endfunction()

set(BUCKYBALL_EXPECTED_COMPILER_DIR "${BUCKYBALL_REPO_ROOT}/compiler")
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
    "Compiler mismatch: workload chip is ${BUCKYBALL_WORKLOAD_CHIP}, "
    "expected BUDDY_EXTERNAL_DIALECTS_DIR ${BUCKYBALL_EXPECTED_COMPILER_DIR}, "
    "but got ${BUCKYBALL_COMPILER_DIR}.")
endif()
file(STRINGS "${BUCKYBALL_COMPILER_CACHE}" BUCKYBALL_COMPILER_PB_LINES
  REGEX "^BUCKYBALL_CHIP_PB:[^=]+=")
list(LENGTH BUCKYBALL_COMPILER_PB_LINES BUCKYBALL_COMPILER_PB_COUNT)
if(NOT BUCKYBALL_COMPILER_PB_COUNT EQUAL 1)
  message(FATAL_ERROR "BUCKYBALL_CHIP_PB is missing or ambiguous in ${BUCKYBALL_COMPILER_CACHE}")
endif()
list(GET BUCKYBALL_COMPILER_PB_LINES 0 BUCKYBALL_COMPILER_PB_LINE)
string(REGEX REPLACE "^BUCKYBALL_CHIP_PB:[^=]+=" ""
  BUCKYBALL_COMPILER_PB "${BUCKYBALL_COMPILER_PB_LINE}")
get_filename_component(BUCKYBALL_COMPILER_PB "${BUCKYBALL_COMPILER_PB}" ABSOLUTE)
get_filename_component(BUCKYBALL_WORKLOAD_PB "${BUCKYBALL_CHIP_PB}" ABSOLUTE)
if(NOT BUCKYBALL_COMPILER_PB STREQUAL BUCKYBALL_WORKLOAD_PB)
  message(FATAL_ERROR
    "Compiler Chip.pb mismatch: compiler uses ${BUCKYBALL_COMPILER_PB}, workload uses ${BUCKYBALL_WORKLOAD_PB}")
endif()

option(BUCKYBALL_STABLE "Use stable LLVM Buckyball extensions" OFF)
if(BUCKYBALL_STABLE)
  set(BUCKYBALL_RISCV_MATTR "+xbbpebble,+xbuckyball,+D")
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

set(BUILD_BIN_DIR ${CMAKE_BINARY_DIR})
set(OUTPUT_BIN_DIR ${WORKLOAD_DIR}/../output/${BUCKYBALL_WORKLOAD_CHIP}/workloads)

#-------------------------------------------------------------------------------
# add_subdirectory
#-------------------------------------------------------------------------------
add_subdirectory(${WORKLOAD_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/lib)
add_subdirectory(src)

add_custom_target(ctest-workloads-build
  DEPENDS
    base-ctests-build
    tlb-ctests-build
    tutorial-ctests-build
    chip-ctests-build
  COMMENT "Building CTest workloads"
)

add_custom_target(mlirtest-workloads-build
  DEPENDS
    MLIRTest-all
    chip-mlirtests-build
  COMMENT "Building MLIRTest workloads"
)

add_custom_target(sync-mlirtest-bin
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src
  COMMAND rsync -a --delete --exclude=CMakeFiles --include="*/"
          --include="MLIRTest/***" --include="balls/***"
          --include="*/mlir_tests/***" --exclude="*"
          ${BUILD_BIN_DIR}/src/ ${OUTPUT_BIN_DIR}/src/
  DEPENDS mlirtest-workloads-build
  COMMENT "Syncing MLIRTest binaries to output directory"
)

add_custom_target(sync-ctest-bin
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/CTest
  COMMAND rsync -a --delete --delete-excluded --exclude=CMakeFiles
          --exclude="*/mlir_tests/**" --include="*/"
          --include="*-baremetal" --include="*-linux" --exclude="*"
          ${BUILD_BIN_DIR}/src/CTest/ ${OUTPUT_BIN_DIR}/src/CTest/
  DEPENDS ctest-workloads-build
  COMMENT "Syncing CTest binaries to output directory"
)

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
          --filter="P_src/MLIRTest/rushB/"
          --filter="P_src/MLIRTest/rushB/***"
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
  tlb-ctests-build
  tutorial-ctests-build
  chip-workloads-build
  MLIRTest-all
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
