set(ELF_CC "riscv64-unknown-elf-gcc")

#-------------------------------------------------------------------------------
# Set baremetal compilation flags
#-------------------------------------------------------------------------------
set(BBSIM_LD ${CTEST_TOY_WORKLOAD_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD} -I${CTEST_TOY_WORKLOAD_DIR})

#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")

# Generate Linux version executables
function(add_linux_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE}
    ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
    ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c)
  target_include_directories(${EXECUTABLE} PRIVATE ${WORKLOAD_LIB_DIR})
  set_target_properties(${EXECUTABLE} PROPERTIES LINKER_LANGUAGE C)
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
    DEPENDS ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
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
      ${CTEST_TOY_WORKLOAD_DIR}/crt0.S
      ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
      ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
    DEPENDS ${CTEST_TOY_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/crt0.S
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
add_cross_platform_test_target(ctest_mvin_mvout_acc_test mvin_mvout_acc_test.c)
add_cross_platform_test_target(ctest_mvin_mvout_test mvin_mvout_test.c)
add_cross_platform_test_target(ctest_vecunit_matmul_ones vecunit_matmul_ones.c)
add_cross_platform_test_target(ctest_vecunit_matmul_identity_random vecunit_matmul_identity_random.c)
add_cross_platform_test_target(ctest_vecunit_matmul_row_col_vector vecunit_matmul_row_col_vector.c)
add_cross_platform_test_target(ctest_vecunit_matmul_col_row_vector vecunit_matmul_col_row_vector.c)
add_cross_platform_test_target(ctest_vecunit_matmul_random1 vecunit_matmul_random1.c)
add_cross_platform_test_target(ctest_vecunit_matmul_random2 vecunit_matmul_random2.c)
add_cross_platform_test_target(ctest_vecunit_tiled_matmul vecunit_tiled_matmul.c)
add_cross_platform_test_target(ctest_vecunit_matmul_zero_random vecunit_matmul_zero_random.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_ones vecunit_matmul_16xn_ones.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random1 vecunit_matmul_16xn_random1.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random2 vecunit_matmul_16xn_random2.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_random3 vecunit_matmul_16xn_random3.c)
add_cross_platform_test_target(ctest_vecunit_matmul_16xn_zero_random vecunit_matmul_16xn_zero_random.c)
add_cross_platform_test_target(ctest_im2col_test im2col_test.c)
add_cross_platform_test_target(ctest_transpose_test transpose_test.c)
add_cross_platform_test_target(ctest_transpose_16xn_test transpose_16xn_test.c)
add_cross_platform_test_target(ctest_transpose_matmul transpose_matmul.c)
add_cross_platform_test_target(ctest_relu_test relu_test.c)
add_cross_platform_test_target(ctest_vsetvli test_vsetvli.c)
add_cross_platform_test_target(ctest_bfp_test bfp_test.c)
add_cross_platform_test_target(ctest_quant_test quant_test.c)
add_cross_platform_test_target(ctest_dequant_test dequant_test.c)
add_cross_platform_test_target(ctest_tlb_test tlb_test.c)
add_cross_platform_test_target(ctest_bdb_counter_test bdb_counter_test.c)
add_cross_platform_test_target(ctest_gemmini_os_risc_basic_test gemmini_os_risc_basic_test.c)
add_cross_platform_test_target(ctest_gemmini_os_risc_shift_test gemmini_os_risc_shift_test.c)
add_cross_platform_test_target(ctest_gemmini_os_risc_atranspose_test gemmini_os_risc_atranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_os_risc_btranspose_test gemmini_os_risc_btranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_os_risc_abtranspose_test gemmini_os_risc_abtranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_ws_risc_basic_test gemmini_ws_risc_basic_test.c)
add_cross_platform_test_target(ctest_gemmini_ws_risc_shift_test gemmini_ws_risc_shift_test.c)
add_cross_platform_test_target(ctest_gemmini_ws_risc_btranspose_test gemmini_ws_risc_btranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_os_cisc_basic_test gemmini_os_cisc_basic_test.c)
add_cross_platform_test_target(ctest_gemmini_os_cisc_shift_test gemmini_os_cisc_shift_test.c)
add_cross_platform_test_target(ctest_gemmini_os_cisc_atranspose_test gemmini_os_cisc_atranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_os_cisc_btranspose_test gemmini_os_cisc_btranspose_test.c)
add_cross_platform_test_target(ctest_gemmini_ws_cisc_conv_test gemmini_ws_cisc_conv_test.c)

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
  ctest_vecunit_tiled_matmul
  ctest_vecunit_matmul_zero_random
  ctest_vecunit_matmul_16xn_ones
  ctest_vecunit_matmul_16xn_random1
  ctest_vecunit_matmul_16xn_random2
  ctest_vecunit_matmul_16xn_random3
  ctest_vecunit_matmul_16xn_zero_random
  ctest_im2col_test
  ctest_transpose_test
  ctest_transpose_16xn_test
  ctest_transpose_matmul
  ctest_relu_test
  ctest_vsetvli
  ctest_bfp_test
  ctest_quant_test
  ctest_dequant_test
  ctest_tlb_test
  ctest_bdb_counter_test
  ctest_gemmini_os_risc_basic_test
  ctest_gemmini_os_risc_shift_test
  ctest_gemmini_os_risc_atranspose_test
  ctest_gemmini_os_risc_btranspose_test
  ctest_gemmini_os_risc_abtranspose_test
  ctest_gemmini_ws_risc_basic_test
  ctest_gemmini_ws_risc_shift_test
  ctest_gemmini_ws_risc_btranspose_test
  ctest_gemmini_os_cisc_basic_test
  ctest_gemmini_os_cisc_shift_test
  ctest_gemmini_os_cisc_atranspose_test
  ctest_gemmini_os_cisc_btranspose_test
  ctest_gemmini_ws_cisc_conv_test
  COMMENT "Building all workloads for Buckyball"
  VERBATIM)
