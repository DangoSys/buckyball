set(RISCV_GNU_TOOLCHAIN ${RISCV})
set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-g++") 

#-------------------------------------------------------------------------------
# 定义公共的编译步骤函数
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 生成不同平台的可执行文件
#-------------------------------------------------------------------------------
# 生成Linux版本的可执行文件
function(add_linux_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")
  
  add_executable(${EXECUTABLE} ${CTEST_WORKLOAD_DIR}/${SOURCE_FILE})
  target_link_libraries(${EXECUTABLE} buckyball-lib)
endfunction()

# 生成multicore baremetal版本的可执行文件
function(add_multicore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")
  
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE} ${CTEST_WORKLOAD_DIR}/start.S
      -DMULTICORE=3
      ${CTEST_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_WORKLOAD_DIR}/buckyball.c
    DEPENDS ${CTEST_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_WORKLOAD_DIR}/start.S
      ${CTEST_WORKLOAD_DIR}/buckyball.c
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_custom_target(${TEST_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

# 生成singlecore baremetal版本的可执行文件
function(add_singlecore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_singlecore-baremetal")
  
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE}
      ${CTEST_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_WORKLOAD_DIR}/buckyball.c
    DEPENDS ${CTEST_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_WORKLOAD_DIR}/buckyball.c
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_custom_target(${TEST_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
  )
endfunction()

# 创建跨平台测试目标
function(add_cross_platform_test_target TEST_NAME SOURCE_FILE)
  add_linux_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_multicore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_singlecore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})
  
  # 创建一个总目标，同时构建所有平台
  add_custom_target(${TEST_NAME}
    DEPENDS ${TEST_NAME}-linux ${TEST_NAME}_multicore_baremetal ${TEST_NAME}_singlecore_baremetal
    COMMENT "Building ${TEST_NAME} for all platforms"
  )
endfunction()

#-------------------------------------------------------------------------------
# build linux version workload (current not supported)
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

add_library(buckyball-lib STATIC ${CTEST_WORKLOAD_DIR}/buckyball.c)

#-------------------------------------------------------------------------------
# 设置baremetal编译标志
#-------------------------------------------------------------------------------
set(C_FLAGS -std=gnu11 -g -fno-common -O2 -static 
  -fno-builtin-printf -specs=htif_nano.specs -I${CTEST_WORKLOAD_DIR})

#-------------------------------------------------------------------------------
# 构建列表 - 使用跨平台function生成所有测试目标
#-------------------------------------------------------------------------------
add_cross_platform_test_target(ctest_mvin_mvout_acc_test mvin_mvout_acc_test.c)
add_cross_platform_test_target(ctest_mvin_mvout_alternate_test mvin_mvout_alternate_test.c)
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


# 创建总的构建目标
add_custom_target(buckyball-CTest-build ALL DEPENDS
  ctest_mvin_mvout_acc_test
  ctest_mvin_mvout_alternate_test
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
  COMMENT "Building all workloads for Buckyball"
  VERBATIM)
