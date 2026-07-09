if(NOT DEFINED ELF_CC)
  set(ELF_CC "riscv64-unknown-elf-gcc")
endif()

if(NOT DEFINED LINUX_CC)
  set(LINUX_CC "riscv64-unknown-linux-gnu-gcc")
endif()

set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(BUCKYBALL_CTEST_C_FLAGS
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
  -I${BUCKYBALL_TOY_COMMON_DIR}
)

set(CMAKE_C_COMPILER ${LINUX_CC})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker")

function(buckyball_ctest_deps OUT_DEPS SOURCE_DIR SOURCE_FILE)
  set(DEPS
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.h)

  if(SOURCE_FILE MATCHES "^tlb_.*\\.c$")
    list(APPEND DEPS ${BUCKYBALL_TOY_COMMON_DIR}/tlb_common.h)
  endif()

  set(${OUT_DEPS} ${DEPS} PARENT_SCOPE)
endfunction()

function(add_buckyball_linux_ctest TEST_NAME SOURCE_DIR SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE}
    ${SOURCE_DIR}/${SOURCE_FILE}
    ${BUCKYBALL_TOY_COMMON_DIR}/buckyball.c)
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
    ${BUCKYBALL_TOY_COMMON_DIR}
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
      ${SOURCE_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
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
      ${SOURCE_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
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

function(add_buckyball_ctest SOURCE_FILE)
  get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_buckyball_linux_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  add_buckyball_multicore_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})
  add_buckyball_singlecore_ctest(${TEST_NAME} ${SOURCE_DIR} ${SOURCE_FILE})

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
