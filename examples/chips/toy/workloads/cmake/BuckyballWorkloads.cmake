if(NOT DEFINED ELF_CC)
  set(ELF_CC "riscv64-unknown-elf-gcc")
endif()

if(NOT DEFINED LINUX_CC)
  set(LINUX_CC "riscv64-unknown-linux-gnu-gcc")
endif()

if(NOT DEFINED BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS)
  set(BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS ${BUCKYBALL_TOY_COMMON_DIR})
endif()

set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(BUDDY_CYCLE_TRACE_RUNTIME_SRC ${BBSW_BAREMETAL_DIR}/buddy_cycle_trace.c)
set(BUCKYBALL_CTEST_C_FLAGS
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
  -I${BUCKYBALL_TOY_COMMON_DIR}
)

set(CMAKE_C_COMPILER ${LINUX_CC})
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

  set(RUSHB_OUTPUT_ROOT ${OUTPUT_BIN_DIR}/${ARG_OUTPUT_SUBDIR})
  string(MAKE_C_IDENTIFIER "${ARG_OUTPUT_SUBDIR}" RUSHB_OUTPUT_ROOT_ID)
  set(RUSHB_OUTPUT_ROOT_TARGET rushB-output-${RUSHB_OUTPUT_ROOT_ID})
  if(NOT TARGET ${RUSHB_OUTPUT_ROOT_TARGET})
    add_custom_target(${RUSHB_OUTPUT_ROOT_TARGET}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_OUTPUT_ROOT}
      COMMENT "Creating rushB output directory: ${ARG_OUTPUT_SUBDIR}"
      VERBATIM)
  endif()

  if(ARG_CXX)
    set(RUSHB_COMPILER c++)
    set(RUSHB_STANDARD -std=c++17)
  else()
    set(RUSHB_COMPILER cc)
    set(RUSHB_STANDARD -std=c17)
  endif()

  foreach(RUSHB_BACKEND bemu verilator)
    set(RUSHB_TARGET ${TARGET_NAME}-rushB-${RUSHB_BACKEND}-run)
    set(RUSHB_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/rushB/${RUSHB_TARGET})
    set(RUSHB_BINARY ${RUSHB_BUILD_DIR}/${RUSHB_TARGET}.bin)
    set(RUSHB_OUTPUT_DIR
      ${RUSHB_OUTPUT_ROOT}/${RUSHB_TARGET})
    if(RUSHB_BACKEND STREQUAL "bemu")
      set(RUSHB_LIBRARY ${BUCKYBALL_RUSHB_BEMU_LIBRARY})
      set(RUSHB_LIBRARY_NAME bebop_bemu)
      set(RUSHB_BUILD_RUNTIME
        COMMAND cargo build --release --manifest-path ${BUCKYBALL_RUSHB_BEMU_MANIFEST} --lib)
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
        "Include examples/chips/<chip>/workloads/cmake/RushB.cmake first.")
    endif()

    set(RUSHB_RUNTIME_OUTPUT
      ${RUSHB_OUTPUT_ROOT}/lib${RUSHB_LIBRARY_NAME}.so)
    set(RUSHB_RUNTIME_TARGET
      rushB-runtime-${RUSHB_OUTPUT_ROOT_ID}-${RUSHB_BACKEND})
    if(NOT TARGET ${RUSHB_RUNTIME_TARGET})
      add_custom_command(
        OUTPUT ${RUSHB_RUNTIME_OUTPUT}
        ${RUSHB_BUILD_RUNTIME}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${RUSHB_LIBRARY} ${RUSHB_RUNTIME_OUTPUT}
        DEPENDS ${RUSHB_RUNTIME_DEPENDENCY}
        COMMENT "Installing rushB ${RUSHB_BACKEND} runtime: ${ARG_OUTPUT_SUBDIR}"
        VERBATIM)
      add_custom_target(${RUSHB_RUNTIME_TARGET} DEPENDS ${RUSHB_RUNTIME_OUTPUT})
      add_dependencies(${RUSHB_RUNTIME_TARGET} ${RUSHB_OUTPUT_ROOT_TARGET})
    endif()

    get_filename_component(RUSHB_LIBRARY_DIR ${RUSHB_LIBRARY} DIRECTORY)
    set(RUSHB_INCLUDE_ARGS -I${BUCKYBALL_REPO_ROOT}/compiler/include)
    foreach(RUSHB_INCLUDE_DIR ${ARG_INCLUDE_DIRS})
      list(APPEND RUSHB_INCLUDE_ARGS -I${RUSHB_INCLUDE_DIR})
    endforeach()
    add_custom_command(
      OUTPUT ${RUSHB_BINARY}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_BUILD_DIR}
      COMMAND ${RUSHB_COMPILER} -no-pie ${RUSHB_STANDARD} -O2 -DBUCKYBALL_RUSHB
              ${RUSHB_INCLUDE_ARGS}
              ${ARG_SOURCES} ${BUCKYBALL_REPO_ROOT}/compiler/lib/RushBRuntime.c
              -L${RUSHB_LIBRARY_DIR} -l${RUSHB_LIBRARY_NAME}
              -Wl,-rpath,${RUSHB_OUTPUT_ROOT} -o ${RUSHB_BINARY}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUSHB_OUTPUT_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUSHB_BINARY}
              ${RUSHB_OUTPUT_DIR}/${RUSHB_TARGET}
      DEPENDS ${ARG_SOURCES} ${ARG_DEPENDS}
              ${BUCKYBALL_REPO_ROOT}/compiler/include/buckyball/rushb.h
              ${BUCKYBALL_REPO_ROOT}/compiler/lib/RushBRuntime.c
      COMMENT "Building rushB ${RUSHB_BACKEND}: ${TARGET_NAME}"
      VERBATIM)
    add_custom_target(${RUSHB_TARGET} DEPENDS ${RUSHB_BINARY})
    add_dependencies(${RUSHB_TARGET} ${RUSHB_OUTPUT_ROOT_TARGET})
    add_dependencies(${RUSHB_TARGET} ${RUSHB_RUNTIME_TARGET})
    add_dependencies(rushB-${RUSHB_BACKEND}-workloads-build ${RUSHB_TARGET})
  endforeach()
endfunction()

set(BUCKYBALL_BBHW_MEM_C ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)

function(buckyball_ctest_deps OUT_DEPS SOURCE_DIR SOURCE_FILE)
  file(GLOB BUCKYBALL_ISA_DEPS CONFIGURE_DEPENDS
    ${WORKLOAD_LIB_DIR}/bbhw/isa/*)
  set(DEPS
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.h
    ${BUCKYBALL_BBHW_MEM_C}
    ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
    ${BUCKYBALL_CHIP_COMMON_SOURCES}
    ${BUCKYBALL_CHIP_COMMON_HEADERS}
    ${BUCKYBALL_ISA_DEPS})

  if(SOURCE_FILE MATCHES "^tlb_.*\\.c$")
    list(APPEND DEPS ${BUCKYBALL_TOY_COMMON_DIR}/tlb_common.h)
  endif()

  set(${OUT_DEPS} ${DEPS} PARENT_SCOPE)
endfunction()

function(add_buckyball_linux_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE}
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
    ${BUCKYBALL_BBHW_MEM_C}
    ${BUCKYBALL_CHIP_COMMON_SOURCES})
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
    ${BUCKYBALL_TOY_COMMON_DIR}
    ${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
    ${SOURCE_DIR})
  set_target_properties(${EXECUTABLE} PROPERTIES LINKER_LANGUAGE C)

  add_custom_target(${TEST_NAME}-linux-build
    DEPENDS ${EXECUTABLE})
endfunction()

function(add_buckyball_multicore_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-multicore-baremetal")
  buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/start.S
      -DMULTICORE=3
      ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
      ${BUCKYBALL_BBHW_MEM_C}
      ${BUCKYBALL_CHIP_COMMON_SOURCES}
      ${SOURCE_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      -I${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
      -I${SOURCE_DIR}
    DEPENDS
      ${TEST_DEPS}
      ${BBSW_BAREMETAL_DIR}/start.S
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-multicore-baremetal-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

function(add_buckyball_singlecore_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-singlecore-baremetal")
  buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
      -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
      ${BUCKYBALL_BBHW_MEM_C}
      ${BUCKYBALL_CHIP_COMMON_SOURCES}
      ${SOURCE_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      -I${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
      -I${SOURCE_DIR}
    DEPENDS
      ${TEST_DEPS}
      ${BBSW_BAREMETAL_DIR}/crt0.S
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-singlecore-baremetal-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

get_filename_component(_BUCKYBALL_CHIPS_CMAKE_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../../cmake" ABSOLUTE)
include(${_BUCKYBALL_CHIPS_CMAKE_DIR}/BuckyballCtestLimits.cmake)

function(add_buckyball_ctest SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR
      "BUCKYBALL_WORKLOAD_CHIP must be set before add_buckyball_ctest()")
  endif()
  get_filename_component(_stem ${SOURCE_FILE} NAME_WE)
  set(TEST_NAME "${BUCKYBALL_WORKLOAD_CHIP}_${_stem}")
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  buckyball_ctest_deps(TEST_DEPS ${SOURCE_DIR} ${SOURCE_FILE})

  add_buckyball_linux_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  add_buckyball_multicore_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  add_buckyball_singlecore_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  if(DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST AND
     DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
    add_buckyball_rushb_native(${TEST_NAME}
      OUTPUT_SUBDIR src/CTest/rushB
      SOURCES
        ${SOURCE_DIR}/${SOURCE_FILE}
        ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
        ${BUCKYBALL_BBHW_MEM_C}
        ${BUCKYBALL_CHIP_COMMON_SOURCES}
      INCLUDE_DIRS
        ${WORKLOAD_LIB_DIR}
        ${BUCKYBALL_TOY_COMMON_DIR}
        ${BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS}
        ${SOURCE_DIR}
      DEPENDS ${TEST_DEPS})
  endif()

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS
      ${TEST_NAME}-linux-build
      ${TEST_NAME}-multicore-baremetal-build
      ${TEST_NAME}-singlecore-baremetal-build
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

function(buckyball_add_ball_ctest_subdirs)
  if(NOT DEFINED BUCKYBALL_BALL_CTEST_DIRS)
    message(FATAL_ERROR "BUCKYBALL_BALL_CTEST_DIRS must be set before buckyball_add_ball_ctest_subdirs()")
  endif()

  foreach(BUCKYBALL_BALL_CTEST_DIR ${BUCKYBALL_BALL_CTEST_DIRS})
    get_filename_component(BUCKYBALL_BALL_WORKLOADS_DIR ${BUCKYBALL_BALL_CTEST_DIR} DIRECTORY)
    get_filename_component(BUCKYBALL_BALL_DIR ${BUCKYBALL_BALL_WORKLOADS_DIR} DIRECTORY)
    get_filename_component(BUCKYBALL_BALL_NAME ${BUCKYBALL_BALL_DIR} NAME)
    add_subdirectory(
      ${BUCKYBALL_BALL_CTEST_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_BALL_NAME}/ctests)
  endforeach()
endfunction()
