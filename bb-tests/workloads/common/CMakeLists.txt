get_filename_component(WORKLOAD_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)

get_filename_component(BUCKYBALL_REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)
include_directories(${BUCKYBALL_REPO_ROOT}/stack/runtime/include)
set(BBSW_BAREMETAL_DIR ${WORKLOAD_LIB_DIR}/bbsw/baremetal)

function(buckyball_ctest_name OUT CORE TEST)
  if("${CORE}" STREQUAL "" OR "${TEST}" STREQUAL "")
    message(FATAL_ERROR "buckyball_ctest_name requires core and test")
  endif()
  set(${OUT} "${BUCKYBALL_WORKLOAD_CHIP}-${CORE}-ctest-${TEST}" PARENT_SCOPE)
endfunction()

function(buckyball_mlirtest_name OUT CORE TEST)
  if("${CORE}" STREQUAL "" OR "${TEST}" STREQUAL "")
    message(FATAL_ERROR "buckyball_mlirtest_name requires core and test")
  endif()
  set(${OUT} "${BUCKYBALL_WORKLOAD_CHIP}-${CORE}-mlirtest-${TEST}" PARENT_SCOPE)
endfunction()

function(buckyball_soctest_name OUT CORE TEST)
  if("${CORE}" STREQUAL "" OR "${TEST}" STREQUAL "")
    message(FATAL_ERROR "buckyball_soctest_name requires core and test")
  endif()
  set(${OUT} "${BUCKYBALL_WORKLOAD_CHIP}-${CORE}-soctest-${TEST}" PARENT_SCOPE)
endfunction()

set(RISCV_GNU_TOOLCHAIN $ENV{RISCV})

set(BUCKYBALL_WORKLOAD_CHIP "" CACHE STRING "Buckyball chip for workloads")
if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
  message(FATAL_ERROR
    "BUCKYBALL_WORKLOAD_CHIP is required. Pass -DBUCKYBALL_WORKLOAD_CHIP=<chip>.")
endif()
set(BUCKYBALL_WORKLOAD_SOURCE_CHIP "${BUCKYBALL_WORKLOAD_CHIP}" CACHE STRING
    "Buckyball chip source tree for workloads")
if(NOT IS_DIRECTORY
    "${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_SOURCE_CHIP}")
  message(FATAL_ERROR
    "Buckyball workload source chip does not exist: ${BUCKYBALL_WORKLOAD_SOURCE_CHIP}")
endif()
message(STATUS
  "Buckyball workload instance: ${BUCKYBALL_WORKLOAD_CHIP} (source: ${BUCKYBALL_WORKLOAD_SOURCE_CHIP})")

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
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${BUCKYBALL_CHIP_PB}"
  "${BUCKYBALL_REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py")
set(BUCKYBALL_ISA_DIR "${CMAKE_BINARY_DIR}/generated/isa")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${BUCKYBALL_REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
    --repo "${BUCKYBALL_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --isa-dir "${BUCKYBALL_ISA_DIR}"
    --print-targets
    --print-target-balls
  RESULT_VARIABLE BUCKYBALL_TARGET_REGISTRY_RESULT
  OUTPUT_VARIABLE BUCKYBALL_TARGETS_RAW
  ERROR_VARIABLE BUCKYBALL_TARGET_REGISTRY_ERROR)
if(NOT BUCKYBALL_TARGET_REGISTRY_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to generate target ISA headers: ${BUCKYBALL_TARGET_REGISTRY_ERROR}")
endif()
string(REPLACE "\n" ";" BUCKYBALL_TARGETS "${BUCKYBALL_TARGETS_RAW}")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${BUCKYBALL_REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
    --repo "${BUCKYBALL_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-bank-targets
  RESULT_VARIABLE BUCKYBALL_BANK_TARGETS_RESULT
  OUTPUT_VARIABLE BUCKYBALL_BANK_TARGETS_RAW
  ERROR_VARIABLE BUCKYBALL_BANK_TARGETS_ERROR)
if(NOT BUCKYBALL_BANK_TARGETS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to get compiler targets with banks: ${BUCKYBALL_BANK_TARGETS_ERROR}")
endif()
string(REPLACE "\n" ";" BUCKYBALL_BANK_TARGETS
  "${BUCKYBALL_BANK_TARGETS_RAW}")
list(REMOVE_ITEM BUCKYBALL_BANK_TARGETS "")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${BUCKYBALL_REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
    --repo "${BUCKYBALL_REPO_ROOT}"
    --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-core-workload-targets
  RESULT_VARIABLE BUCKYBALL_CORE_TARGETS_RESULT
  OUTPUT_VARIABLE BUCKYBALL_CORE_TARGETS_RAW
  ERROR_VARIABLE BUCKYBALL_CORE_TARGETS_ERROR)
if(NOT BUCKYBALL_CORE_TARGETS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "failed to get Core workload targets: ${BUCKYBALL_CORE_TARGETS_ERROR}")
endif()
string(REPLACE "\n" ";" BUCKYBALL_CORE_TARGETS
  "${BUCKYBALL_CORE_TARGETS_RAW}")
list(REMOVE_ITEM BUCKYBALL_CORE_TARGETS "")
set(BUCKYBALL_TARGET_BALL_LINES)
foreach(BUCKYBALL_TARGET_LINE ${BUCKYBALL_TARGETS})
  if(BUCKYBALL_TARGET_LINE MATCHES "^([^:]+):(.*)$")
    set(BUCKYBALL_TARGET_BALLS_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    list(APPEND BUCKYBALL_TARGET_BALL_LINES ${BUCKYBALL_TARGET_LINE})
  endif()
endforeach()
list(REMOVE_ITEM BUCKYBALL_TARGETS ${BUCKYBALL_TARGET_BALL_LINES} "")
if(NOT BUCKYBALL_TARGETS)
  message(FATAL_ERROR "Chip.pb has no compiler targets")
endif()
foreach(_BUCKYBALL_TARGET IN LISTS BUCKYBALL_TARGETS)
  if(NOT EXISTS "${BUCKYBALL_ISA_DIR}/${_BUCKYBALL_TARGET}/params.h")
    message(FATAL_ERROR
      "missing generated params.h for target ${_BUCKYBALL_TARGET}")
  endif()
endforeach()
message(STATUS "Buckyball compiler targets: ${BUCKYBALL_TARGETS}")
message(STATUS "Buckyball bank targets: ${BUCKYBALL_BANK_TARGETS}")
message(STATUS "Buckyball Core workload targets: ${BUCKYBALL_CORE_TARGETS}")
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${BUCKYBALL_REPO_ROOT}/stack/compiler/scripts/pb_to_target_registry.py"
    --repo "${BUCKYBALL_REPO_ROOT}" --chip-pb "${BUCKYBALL_CHIP_PB}"
    --print-cpu-targets
  RESULT_VARIABLE _cpu_result OUTPUT_VARIABLE _cpu_targets ERROR_VARIABLE _cpu_error)
if(NOT _cpu_result EQUAL 0)
  message(FATAL_ERROR "failed to get CPU workload targets: ${_cpu_error}")
endif()
string(REPLACE "\n" ";" BUCKYBALL_CPU_TARGETS "${_cpu_targets}")
list(REMOVE_ITEM BUCKYBALL_CPU_TARGETS "")

function(buckyball_assign_physical_banks TARGET OUT)
  list(FIND BUCKYBALL_BANK_TARGETS "${TARGET}" _bank_target_index)
  if(_bank_target_index EQUAL -1)
    message(FATAL_ERROR "${TARGET}: no physical banks")
  endif()
  set(${OUT} "-assign-physical-banks" PARENT_SCOPE)
endfunction()

set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker")

function(buckyball_chip_ball_dirs OUT_VAR TARGET)
  set(_raw "${BUCKYBALL_TARGET_BALLS_${TARGET}}")
  if(_raw STREQUAL "")
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  string(REPLACE "," ";" _dirs "${_raw}")
  foreach(_dir IN LISTS _dirs)
    if(_dir STREQUAL "")
      message(FATAL_ERROR "empty ball_dir on target ${TARGET}")
    endif()
  endforeach()
  set(${OUT_VAR} "${_dirs}" PARENT_SCOPE)
endfunction()

function(buckyball_add_ball_workload_subdirs KIND)
  if(NOT KIND STREQUAL "ctests" AND NOT KIND STREQUAL "mlir_tests")
    message(FATAL_ERROR "unknown ball workload kind: ${KIND}")
  endif()
  foreach(BUCKYBALL_TARGET IN LISTS BUCKYBALL_TARGETS)
    if(NOT BUCKYBALL_TARGET IN_LIST BUCKYBALL_CPU_TARGETS)
      continue()
    endif()
    buckyball_chip_ball_dirs(_ball_dirs ${BUCKYBALL_TARGET})
    foreach(BUCKYBALL_BALL_DIR IN LISTS _ball_dirs)
      set(_src
        ${BUCKYBALL_REPO_ROOT}/examples/balls/${BUCKYBALL_BALL_DIR}/workloads/${KIND})
      if(NOT IS_DIRECTORY ${_src})
        if(KIND STREQUAL "mlir_tests")
          continue()
        endif()
        message(FATAL_ERROR "missing Ball ${KIND} directory: ${_src}")
      endif()
      if(KIND STREQUAL "ctests")
        set(BUCKYBALL_CTEST_TARGET ${BUCKYBALL_TARGET})
        add_subdirectory(
          ${_src}
          ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_TARGET}/${BUCKYBALL_BALL_DIR}/ctests)
      else()
        set(BUCKYBALL_MLIR_ACTIVE_TARGET ${BUCKYBALL_TARGET})
        add_subdirectory(
          ${_src}
          ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_TARGET}/${BUCKYBALL_BALL_DIR})
      endif()
    endforeach()
  endforeach()
endfunction()

function(buckyball_add_core_workload_subdirs KIND)
  foreach(_entry IN LISTS BUCKYBALL_CORE_TARGETS)
    if(NOT _entry MATCHES "^([^:]+):([^:]+):([0-9]+)$")
      message(FATAL_ERROR "invalid Core workload target: ${_entry}")
    endif()
    set(BUCKYBALL_CORE_WORKLOAD_TARGET "${CMAKE_MATCH_1}")
    if(KIND STREQUAL "ctests" AND NOT BUCKYBALL_CORE_WORKLOAD_TARGET IN_LIST BUCKYBALL_CPU_TARGETS)
      continue()
    endif()
    set(_pkg "${CMAKE_MATCH_2}")
    set(BUCKYBALL_CORE_WORKLOAD_PROFILE "${CMAKE_MATCH_3}")
    set(_src
      "${BUCKYBALL_REPO_ROOT}/examples/cores/${_pkg}/workloads/${KIND}")
    if(IS_DIRECTORY "${_src}")
      add_subdirectory("${_src}"
        "${CMAKE_CURRENT_BINARY_DIR}/cores/${BUCKYBALL_CORE_WORKLOAD_TARGET}/${KIND}")
    endif()
  endforeach()
endfunction()

set(ELF_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc)
set(LINUX_CC ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
set(BUILD_BIN_DIR ${CMAKE_BINARY_DIR})
set(OUTPUT_BIN_DIR ${WORKLOAD_DIR}/../output/${BUCKYBALL_WORKLOAD_CHIP}/workloads)
