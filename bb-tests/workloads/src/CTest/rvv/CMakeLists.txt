set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

#-------------------------------------------------------------------------------
# Set baremetal compilation flags with RVV support
#-------------------------------------------------------------------------------
set(C_FLAGS -g -fno-common -O1 -static -march=rv64gcv -mabi=lp64d -mcmodel=medany
  -fno-builtin-printf -specs=htif_nano.specs)

#-------------------------------------------------------------------------------
# Define common compilation step functions
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")

# Generate Linux version executables
function(add_linux_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE} ${CTEST_RVV_WORKLOAD_DIR}/${SOURCE_FILE})
  target_compile_options(${EXECUTABLE} PRIVATE -march=rv64gcv -mabi=lp64d)
endfunction()

# Generate multicore baremetal version executables
function(add_multicore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE} ${CTEST_RVV_WORKLOAD_DIR}/start.S
      -DMULTICORE=3
      ${CTEST_RVV_WORKLOAD_DIR}/${SOURCE_FILE}
    DEPENDS ${CTEST_RVV_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_RVV_WORKLOAD_DIR}/start.S
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

# Generate singlecore baremetal version executables
function(add_singlecore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE}
      ${CTEST_RVV_WORKLOAD_DIR}/start.S
      ${CTEST_RVV_WORKLOAD_DIR}/${SOURCE_FILE}
    DEPENDS ${CTEST_RVV_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_RVV_WORKLOAD_DIR}/start.S
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

# Create cross-platform test targets
function(add_cross_platform_test_target TEST_NAME SOURCE_FILE)
  add_linux_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_multicore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_singlecore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})

  # Create a master target that builds all platforms simultaneously
  add_custom_target(${TEST_NAME}
    DEPENDS ${TEST_NAME}-linux ${TEST_NAME}_multicore_baremetal ${TEST_NAME}_singlecore_baremetal
    COMMENT "Building ${TEST_NAME} for all platforms"
  )
endfunction()

#-------------------------------------------------------------------------------
# build linux version workload
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

#-------------------------------------------------------------------------------
# Build list - use cross-platform function to generate all test targets
#-------------------------------------------------------------------------------
add_cross_platform_test_target(ctest_rvv_vadd vadd_test.c)
add_cross_platform_test_target(ctest_rvv_vdot vdot_test.c)

# Create master build target
add_custom_target(buckyball-rvv-build ALL DEPENDS
  ctest_rvv_vadd
  ctest_rvv_vdot
  COMMENT "Building all RVV workloads for Buckyball"
  VERBATIM)
