set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

#-------------------------------------------------------------------------------
# Set baremetal compilation flags
#-------------------------------------------------------------------------------
set(BBSIM_LD ${CTEST_TOY_WORKLOAD_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD} -I${CTEST_TOY_WORKLOAD_DIR})

#-------------------------------------------------------------------------------
# Define common compilation step functions
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc")
# Generate Linux version executables
function(add_linux_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE} ${CTEST_BEBOP_WORKLOAD_DIR}/${SOURCE_FILE} ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c)
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
  )
endfunction()

# Generate multicore baremetal version executables
function(add_multicore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE} ${CTEST_TOY_WORKLOAD_DIR}/start.S
      -DMULTICORE=3
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CTEST_BEBOP_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
    DEPENDS ${CTEST_BEBOP_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/start.S
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

# Generate singlecore baremetal version executables
function(add_singlecore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE}
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CTEST_BEBOP_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
    DEPENDS ${CTEST_BEBOP_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

# Create cross-platform test targets
function(add_cross_platform_test_target TEST_NAME SOURCE_FILE)
  add_linux_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_multicore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_singlecore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})

  # Create a master target that builds all platforms simultaneously
  add_custom_target(${TEST_NAME}
    DEPENDS  ${TEST_NAME}-linux ${TEST_NAME}_multicore_baremetal ${TEST_NAME}_singlecore_baremetal
    COMMENT "Building ${TEST_NAME} for all platforms"
  )
endfunction()

#-------------------------------------------------------------------------------
# build linux version workload (current not supported)
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

# add_library(buckyball-lib STATIC ${CTEST_WORKLOAD_DIR}/buckyball.c)
# target_include_directories(buckyball-lib PRIVATE ${WORKLOAD_LIB_DIR})

#-------------------------------------------------------------------------------
# Build list - use cross-platform function to generate all test targets
#-------------------------------------------------------------------------------
add_cross_platform_test_target(ctest_mvin_mvout_bebop_test mvin_mvout_test.c)
add_cross_platform_test_target(ctest_mvin_mvout_mset_bebop_test mvin_mvout_mset_test.c)
add_cross_platform_test_target(ctest_mvin_mvout_acc_bebop_test mvin_mvout_acc_test.c)

# Create master build target
add_custom_target(buckyball-CTest-bebop-build ALL DEPENDS
  ctest_mvin_mvout_bebop_test
  ctest_mvin_mvout_mset_bebop_test
  ctest_mvin_mvout_acc_bebop_test
  COMMENT "Building all workloads for Buckyball"
  VERBATIM)
