# ===============================================================
# Generic functions to help add CTest workloads
# ===============================================================
set(CTEST_MAX_LINES 100)
set(BBHW_MEM_C ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)
set(BUCKYBALL_CTEST_C_FLAGS
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
)
file(GLOB BUCKYBALL_BALL_WORKLOAD_DIRS CONFIGURE_DEPENDS
  "${BUCKYBALL_REPO_ROOT}/examples/balls/*/workloads")
file(GLOB BUCKYBALL_BALL_ISA_DEPS CONFIGURE_DEPENDS
  "${BUCKYBALL_REPO_ROOT}/examples/balls/*/workloads/isa/*")

function(buckyball_enforce_ctest_line_limit SOURCE_FILE)
  if(IS_ABSOLUTE "${SOURCE_FILE}")
    set(_src "${SOURCE_FILE}")
  else()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
  endif()
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "ctest source does not exist: ${_src}")
  endif()

  file(READ "${_src}" _content)
  if(_content STREQUAL "")
    set(_n 0)
  else()
    string(REGEX MATCHALL "\n" _newlines "${_content}")
    list(LENGTH _newlines _n)
    string(REGEX MATCH "[^\n]$" _no_trailing_nl "${_content}")
    if(_no_trailing_nl)
      math(EXPR _n "${_n} + 1")
    endif()
  endif()

  if(_n GREATER CTEST_MAX_LINES)
    message(FATAL_ERROR
      "ctest exceeds ${CTEST_MAX_LINES} lines: ${_src} (${_n} lines). "
      "Split it into smaller focused ctests. "
      "Do not evade this check by moving functional code into headers.")
  endif()
endfunction()

function(buckyball_ctest_deps OUT_DEPS SOURCE_DIR SOURCE_FILE)
  if(NOT DEFINED BUCKYBALL_CTEST_TARGET OR BUCKYBALL_CTEST_TARGET STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_CTEST_TARGET is required for C tests")
  endif()
  list(FIND BUCKYBALL_TARGETS "${BUCKYBALL_CTEST_TARGET}" _buckyball_target_index)
  if(_buckyball_target_index EQUAL -1)
    message(FATAL_ERROR "unknown C test target: ${BUCKYBALL_CTEST_TARGET}")
  endif()
  file(GLOB BUCKYBALL_ISA_DEPS CONFIGURE_DEPENDS
    ${WORKLOAD_LIB_DIR}/bbhw/isa/*)
  set(DEPS
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${WORKLOAD_LIB_DIR}/buckyball.h
    ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h
    ${BBHW_MEM_C}
    ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
    ${BUCKYBALL_CHIP_COMMON_SOURCES}
    ${BUCKYBALL_CHIP_COMMON_HEADERS}
    ${BUCKYBALL_ISA_DEPS}
    ${BUCKYBALL_BALL_ISA_DEPS})
  file(GLOB _LOCAL_HDRS CONFIGURE_DEPENDS ${SOURCE_DIR}/*.h)
  list(APPEND DEPS ${_LOCAL_HDRS})

  set(${OUT_DEPS} ${DEPS} PARENT_SCOPE)
endfunction()

function(add_buckyball_linux_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE}
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${BBHW_MEM_C}
    ${BUCKYBALL_CHIP_COMMON_SOURCES})
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
    ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
    ${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
    ${BUCKYBALL_BALL_WORKLOAD_DIRS}
    ${SOURCE_DIR})
  set_target_properties(${EXECUTABLE} PROPERTIES LINKER_LANGUAGE C)

  add_custom_target(${TEST_NAME}-linux-build
    DEPENDS ${EXECUTABLE})
endfunction()

function(add_buckyball_baremetal_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-baremetal")
  buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

  set(BUCKYBALL_CHIP_COMMON_INCLUDE_FLAGS)
  foreach(CHIP_COMMON_INCLUDE_DIR ${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS})
    list(APPEND BUCKYBALL_CHIP_COMMON_INCLUDE_FLAGS -I${CHIP_COMMON_INCLUDE_DIR})
  endforeach()
  set(BALL_ISA_INCLUDE_FLAGS)
  foreach(BALL_WORKLOAD_DIR ${BUCKYBALL_BALL_WORKLOAD_DIRS})
    list(APPEND BALL_ISA_INCLUDE_FLAGS -I${BALL_WORKLOAD_DIR})
  endforeach()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${WORKLOAD_LIB_DIR}/buckyball.c
      ${BBHW_MEM_C}
      ${BUCKYBALL_CHIP_COMMON_SOURCES}
      ${SOURCE_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
      ${BUCKYBALL_CHIP_COMMON_INCLUDE_FLAGS}
      ${BALL_ISA_INCLUDE_FLAGS}
      -I${SOURCE_DIR}
    DEPENDS
      ${TEST_DEPS}
      ${BBSW_BAREMETAL_DIR}/crt0.S
    COMMENT "Building baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-baremetal-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

function(add_buckyball_ctest SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR
      "BUCKYBALL_WORKLOAD_CHIP must be set before add_buckyball_ctest()")
  endif()
  get_filename_component(_stem ${SOURCE_FILE} NAME_WE)
  if(NOT DEFINED BUCKYBALL_CTEST_TARGET OR BUCKYBALL_CTEST_TARGET STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_CTEST_TARGET is required for ${SOURCE_FILE}")
  endif()
  set(TEST_NAME "${BUCKYBALL_WORKLOAD_CHIP}-${BUCKYBALL_CTEST_TARGET}-ctest-${_stem}")
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

  add_buckyball_linux_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  add_buckyball_baremetal_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  set(_rushb_ok TRUE)
  if(_stem MATCHES "^tlb_" OR _stem STREQUAL "test_vsetvli")
    set(_rushb_ok FALSE)
  endif()
  if(_rushb_ok AND
     DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST AND
     DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
    buckyball_rushb_default_accelerator(
      _rushb_accelerator "${BUCKYBALL_CTEST_TARGET}")
    add_buckyball_rushb_native(${TEST_NAME}
      OUTPUT_SUBDIR src/CTest/rushB
      ACCELERATOR_ID ${_rushb_accelerator}
      SOURCES
        ${SOURCE_DIR}/${SOURCE_FILE}
        ${WORKLOAD_LIB_DIR}/buckyball.c
        ${BBHW_MEM_C}
        ${BUCKYBALL_CHIP_COMMON_SOURCES}
      INCLUDE_DIRS
        ${WORKLOAD_LIB_DIR}
        ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
        ${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
        ${BUCKYBALL_BALL_WORKLOAD_DIRS}
        ${SOURCE_DIR}
      DEPENDS ${TEST_DEPS})
  endif()

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS
      ${TEST_NAME}-linux-build
      ${TEST_NAME}-baremetal-build
    COMMENT "Building CTest workload ${TEST_NAME}"
  )

  if(DEFINED BUCKYBALL_CTEST_GROUP_TARGET)
    add_dependencies(${BUCKYBALL_CTEST_GROUP_TARGET} ${TEST_NAME}-ctest-build)
  endif()
endfunction()

function(add_buckyball_ctests)
  foreach(SOURCE_FILE ${ARGV})
    add_buckyball_ctest(${SOURCE_FILE})
  endforeach()
endfunction()

# ===============================================================
# Common workloads for all chips
# ===============================================================
add_subdirectory(base)
add_subdirectory(tlb)
add_subdirectory(tutorial)
add_subdirectory(embench)
add_subdirectory(coremark)

# ===============================================================
# Chip-specific workloads
# ===============================================================
set(BUCKYBALL_CHIP_WORKLOAD_DIR
  ${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}/workloads)

if(NOT EXISTS ${BUCKYBALL_CHIP_WORKLOAD_DIR}/CMakeLists.txt)
  message(FATAL_ERROR
    "Buckyball chip workloads do not exist: ${BUCKYBALL_CHIP_WORKLOAD_DIR}")
endif()

add_subdirectory(${BUCKYBALL_CHIP_WORKLOAD_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/chips/${BUCKYBALL_WORKLOAD_CHIP})

if(NOT TARGET chip-workloads-build)
  message(FATAL_ERROR
    "Buckyball chip workloads must define target chip-workloads-build: ${BUCKYBALL_WORKLOAD_CHIP}")
endif()
