set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

#-------------------------------------------------------------------------------
# Set baremetal compilation flags
#-------------------------------------------------------------------------------
set(C_FLAGS -g -fno-common -O1 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=htif_nano.specs -I${CTEST_TOY_WORKLOAD_DIR})

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

  add_executable(${EXECUTABLE} ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE} ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c)
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
  )
  # Ensure dependent libraries are built first and link merged library files
  add_dependencies(${EXECUTABLE} bbhw-linux)
  target_link_libraries(${EXECUTABLE} ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-linux.a)
endfunction()

# Generate multicore baremetal version executables
function(add_multicore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE} ${CTEST_TOY_WORKLOAD_DIR}/start.S
      -DMULTICORE=3
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-baremetal.a
    DEPENDS ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/start.S
      ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-baremetal.a
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE} bbhw-baremetal  )
endfunction()

# Generate singlecore baremetal version executables
function(add_singlecore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE}
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-baremetal.a
    DEPENDS ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-baremetal.a
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE} bbhw-baremetal
  )
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
add_cross_platform_test_target(ctest_mvin_mvout_acc_test mvin_mvout_acc_test.c)
add_cross_platform_test_target(ctest_mvin_mvout_test mvin_mvout_test.c)
add_cross_platform_test_target(ctest_vecunit_matmul_ones vecunit_matmul_ones.c)
add_cross_platform_test_target(ctest_vecunit_matmul_identity_random vecunit_matmul_identity_random.c)
add_cross_platform_test_target(ctest_vecunit_matmul_row_col_vector vecunit_matmul_row_col_vector.c)
add_cross_platform_test_target(ctest_vecunit_matmul_col_row_vector vecunit_matmul_col_row_vector.c)
add_cross_platform_test_target(ctest_vecunit_matmul_random1 vecunit_matmul_random1.c)
add_cross_platform_test_target(ctest_vecunit_matmul_random2 vecunit_matmul_random2.c)
add_cross_platform_test_target(ctest_vecunit_matmul_random3 vecunit_matmul_random3.c)
add_cross_platform_test_target(ctest_vecunit_matmul_zero_random vecunit_matmul_zero_random.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_ones vecunit_matmul_16xn_ones.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random1 vecunit_matmul_16xn_random1.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random2 vecunit_matmul_16xn_random2.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random3 vecunit_matmul_16xn_random3.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_zero_random vecunit_matmul_16xn_zero_random.c)
add_cross_platform_test_target(ctest_vecunit_simple_nn_forward_pass_test vecunit_simple_nn_forward_pass_test.c)
add_cross_platform_test_target(ctest_im2col_test im2col_test.c)
add_cross_platform_test_target(ctest_bbfp_matmul_test bbfpmatmul.c)
add_cross_platform_test_target(ctest_bbfp_matmul_random1 bbfp_matmul_random1.c)
add_cross_platform_test_target(ctest_bbfp_matmul_random2 bbfp_matmul_random2.c)
add_cross_platform_test_target(ctest_bbfp_matmul_random3 bbfp_matmul_random3.c)
add_cross_platform_test_target(ctest_transpose_test transpose_test.c)
add_cross_platform_test_target(ctest_transpose_matmul transpose_matmul.c)
add_cross_platform_test_target(ctest_tiled_matmul tiled_matmul.c)
add_cross_platform_test_target(ctest_relu_test relu_test.c)
add_cross_platform_test_target(ctest_nnlut_test nnlut_test.c)
add_cross_platform_test_target(ctest_snn_test snn_test.c)
add_cross_platform_test_target(ctest_abft_systolic_test abft_systolic_test.c)
add_cross_platform_test_target(ctest_conv_test conv_test.c)
add_cross_platform_test_target(ctest_cim_test cim_test.c)
add_cross_platform_test_target(ctest_transfer_test transfer_test.c)
add_cross_platform_test_target(ctest_vsetvli test_vsetvli.c)

# Create library dependency target to ensure all libraries are built first
# add_custom_target(buckyball-libs-build
#   DEPENDS bbmem-linux-build bbisa-linux-build bbmem-baremetal-build bbisa-baremetal-build
#   COMMENT "Building all required libraries"
# )

# Create master build target
add_custom_target(buckyball-CTest-build ALL DEPENDS
  ctest_mvin_mvout_acc_test
  ctest_mvin_mvout_test
  ctest_vecunit_matmul_ones
  ctest_vecunit_matmul_identity_random
  ctest_vecunit_matmul_row_col_vector
  ctest_vecunit_matmul_col_row_vector
  ctest_vecunit_matmul_random1
  ctest_vecunit_matmul_random2
  ctest_vecunit_matmul_random3
  ctest_vecunit_matmul_zero_random
  ctest_vecunit_matmul_16xn_ones
  ctest_vecunit_matmul_16xn_random1
  ctest_vecunit_matmul_16xn_random2
  ctest_vecunit_matmul_16xn_random3
  ctest_vecunit_matmul_16xn_zero_random
  ctest_vecunit_simple_nn_forward_pass_test
  ctest_im2col_test
  ctest_bbfp_matmul_test
  ctest_bbfp_matmul_random1
  ctest_bbfp_matmul_random2
  ctest_bbfp_matmul_random3
  ctest_transpose_test
  ctest_transpose_matmul
  ctest_tiled_matmul
  ctest_relu_test
  ctest_nnlut_test
  ctest_snn_test
  ctest_abft_systolic_test
  ctest_conv_test
  ctest_cim_test
  ctest_transfer_test
  ctest_vsetvli
  COMMENT "Building all workloads for Buckyball"
  VERBATIM)
