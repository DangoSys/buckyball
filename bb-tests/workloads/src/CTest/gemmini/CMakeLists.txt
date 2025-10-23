set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

#-------------------------------------------------------------------------------
# 设置baremetal编译标志
#-------------------------------------------------------------------------------
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -std=gnu99 -ffast-math -fno-builtin-printf -fno-tree-loop-distribute-patterns
  -DBAREMETAL=1 -DPREALLOCATE=1 -DMULTITHREAD=1 -DPRINT_TILE=0
  -specs=htif_nano.specs -I${CTEST_GEMMINI_WORKLOAD_DIR})

# 链接库
set(LINK_LIBS -lm -lgcc)

#-------------------------------------------------------------------------------
# 定义公共的编译步骤函数
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 生成不同平台的可执行文件
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -DPREALLOCATE=1 -DMULTITHREAD=1 -DPRINT_TILE=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc -mcmodel=medany -std=gnu++11 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -DPREALLOCATE=1 -DMULTITHREAD=1 -DPRINT_TILE=0")

# 生成Linux版本的可执行文件
function(add_linux_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}-linux")

  add_executable(${EXECUTABLE} ${CTEST_GEMMINI_WORKLOAD_DIR}/${SOURCE_FILE})
  target_include_directories(${EXECUTABLE} PRIVATE
    ${WORKLOAD_LIB_DIR}
    ${CTEST_GEMMINI_WORKLOAD_DIR}
  )
  # 确保依赖库先构建，并链接合并的库文件
  add_dependencies(${EXECUTABLE} bbhw-linux)
  target_link_libraries(${EXECUTABLE}
    m
  )
endfunction()

# 生成multicore baremetal版本的可执行文件
function(add_multicore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE} ${CTEST_TOY_WORKLOAD_DIR}/start.S
      -DMULTICORE=3
      ${CTEST_GEMMINI_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      ${LINK_LIBS}
    DEPENDS ${CTEST_GEMMINI_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_TOY_WORKLOAD_DIR}/start.S
    COMMENT "Building multicore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE} bbhw-baremetal)
endfunction()

# 生成singlecore baremetal版本的可执行文件
function(add_singlecore_baremetal_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${C_FLAGS} -o ${EXECUTABLE}
      ${CTEST_GEMMINI_WORKLOAD_DIR}/${SOURCE_FILE}
      -I${WORKLOAD_LIB_DIR}
      ${LINK_LIBS}
    DEPENDS ${CTEST_GEMMINI_WORKLOAD_DIR}/${SOURCE_FILE}
    COMMENT "Building singlecore baremetal executable: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE} bbhw-baremetal
  )
endfunction()

# 创建跨平台测试目标
function(add_cross_platform_test_target TEST_NAME SOURCE_FILE)
  add_linux_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_multicore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})
  add_singlecore_baremetal_test_target(${TEST_NAME} ${SOURCE_FILE})

  # 创建一个总目标，同时构建所有平台
  add_custom_target(${TEST_NAME}
    DEPENDS  ${TEST_NAME}-linux ${TEST_NAME}_multicore_baremetal ${TEST_NAME}_singlecore_baremetal
    COMMENT "Building ${TEST_NAME} for all platforms"
  )
endfunction()

#-------------------------------------------------------------------------------
# build linux version workload
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

#-------------------------------------------------------------------------------
# 构建列表 - 使用跨平台function生成所有测试目标
#-------------------------------------------------------------------------------
add_cross_platform_test_target(gemmini_mvin_mvout mvin_mvout.c)
add_cross_platform_test_target(gemmini_mvin_mvout_zeros mvin_mvout_zeros.c)
add_cross_platform_test_target(gemmini_mvin_mvout_stride mvin_mvout_stride.c)
add_cross_platform_test_target(gemmini_mvin_mvout_block_stride mvin_mvout_block_stride.c)
add_cross_platform_test_target(gemmini_mvin_mvout_acc mvin_mvout_acc.c)
add_cross_platform_test_target(gemmini_mvin_mvout_acc_zero_stride mvin_mvout_acc_zero_stride.c)
add_cross_platform_test_target(gemmini_mvin_mvout_acc_stride mvin_mvout_acc_stride.c)
add_cross_platform_test_target(gemmini_mvin_mvout_acc_full mvin_mvout_acc_full.c)
add_cross_platform_test_target(gemmini_mvin_mvout_acc_full_stride mvin_mvout_acc_full_stride.c)
add_cross_platform_test_target(gemmini_matmul_os matmul_os.c)
add_cross_platform_test_target(gemmini_matmul_ws matmul_ws.c)
add_cross_platform_test_target(gemmini_matmul matmul.c)
add_cross_platform_test_target(gemmini_raw_hazard raw_hazard.c)
add_cross_platform_test_target(gemmini_aligned aligned.c)
add_cross_platform_test_target(gemmini_padded padded.c)
add_cross_platform_test_target(gemmini_mvin_scale mvin_scale.c)
add_cross_platform_test_target(gemmini_conv conv.c)
add_cross_platform_test_target(gemmini_conv_stride conv_stride.c)
add_cross_platform_test_target(gemmini_conv_rect conv_rect.c)
add_cross_platform_test_target(gemmini_conv_rect_pool conv_rect_pool.c)
add_cross_platform_test_target(gemmini_conv_with_pool conv_with_pool.c)
add_cross_platform_test_target(gemmini_conv_with_rot180 conv_with_rot180.c)
add_cross_platform_test_target(gemmini_conv_with_kernel_dilation conv_with_kernel_dilation.c)
add_cross_platform_test_target(gemmini_conv_with_input_dilation conv_with_input_dilation.c)
add_cross_platform_test_target(gemmini_conv_with_input_dilation_and_rot180 conv_with_input_dilation_and_rot180.c)
add_cross_platform_test_target(gemmini_conv_with_input_dilation_and_neg_padding conv_with_input_dilation_and_neg_padding.c)
add_cross_platform_test_target(gemmini_conv_trans_output_1203 conv_trans_output_1203.c)
add_cross_platform_test_target(gemmini_conv_trans_weight_1203 conv_trans_weight_1203.c)
add_cross_platform_test_target(gemmini_conv_trans_weight_0132 conv_trans_weight_0132.c)
add_cross_platform_test_target(gemmini_conv_trans_input_3120 conv_trans_input_3120.c)
add_cross_platform_test_target(gemmini_conv_trans_input_3120_with_kernel_dilation conv_trans_input_3120_with_kernel_dilation.c)
add_cross_platform_test_target(gemmini_conv_first_layer conv_first_layer.c)
add_cross_platform_test_target(gemmini_conv_dw conv_dw.c)
add_cross_platform_test_target(gemmini_conv_perf conv_perf.c)
add_cross_platform_test_target(gemmini_conv_dw_perf conv_dw_perf.c)
add_cross_platform_test_target(gemmini_tiled_matmul_os tiled_matmul_os.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws tiled_matmul_ws.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_At tiled_matmul_ws_At.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_Bt tiled_matmul_ws_Bt.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_full_C tiled_matmul_ws_full_C.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_low_D tiled_matmul_ws_low_D.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_igelu tiled_matmul_ws_igelu.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_layernorm tiled_matmul_ws_layernorm.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_softmax tiled_matmul_ws_softmax.c)
add_cross_platform_test_target(gemmini_tiled_matmul_ws_perf tiled_matmul_ws_perf.c)
add_cross_platform_test_target(gemmini_tiled_matmul_cpu tiled_matmul_cpu.c)
add_cross_platform_test_target(gemmini_tiled_matmul_option tiled_matmul_option.c)
add_cross_platform_test_target(gemmini_transpose transpose.c)
add_cross_platform_test_target(gemmini_matrix_add matrix_add.c)
add_cross_platform_test_target(gemmini_resadd resadd.c)
add_cross_platform_test_target(gemmini_resadd_stride resadd_stride.c)
add_cross_platform_test_target(gemmini_global_average global_average.c)
add_cross_platform_test_target(gemmini_gemmini_counter gemmini_counter.c)
add_cross_platform_test_target(gemmini_template template.c)

# 创建总的构建目标
add_custom_target(buckyball-gemmini-build ALL DEPENDS
  gemmini_mvin_mvout
  gemmini_mvin_mvout_zeros
  gemmini_mvin_mvout_stride
  gemmini_mvin_mvout_block_stride
  gemmini_mvin_mvout_acc
  gemmini_mvin_mvout_acc_zero_stride
  gemmini_mvin_mvout_acc_stride
  gemmini_mvin_mvout_acc_full
  gemmini_mvin_mvout_acc_full_stride
  gemmini_matmul_os
  gemmini_matmul_ws
  gemmini_matmul
  gemmini_raw_hazard
  gemmini_aligned
  gemmini_padded
  gemmini_mvin_scale
  gemmini_conv
  gemmini_conv_stride
  gemmini_conv_rect
  gemmini_conv_rect_pool
  gemmini_conv_with_pool
  gemmini_conv_with_rot180
  gemmini_conv_with_kernel_dilation
  gemmini_conv_with_input_dilation
  gemmini_conv_with_input_dilation_and_rot180
  gemmini_conv_with_input_dilation_and_neg_padding
  gemmini_conv_trans_output_1203
  gemmini_conv_trans_weight_1203
  gemmini_conv_trans_weight_0132
  gemmini_conv_trans_input_3120
  gemmini_conv_trans_input_3120_with_kernel_dilation
  gemmini_conv_first_layer
  gemmini_conv_dw
  gemmini_conv_perf
  gemmini_conv_dw_perf
  gemmini_tiled_matmul_os
  gemmini_tiled_matmul_ws
  gemmini_tiled_matmul_ws_At
  gemmini_tiled_matmul_ws_Bt
  gemmini_tiled_matmul_ws_full_C
  gemmini_tiled_matmul_ws_low_D
  gemmini_tiled_matmul_ws_igelu
  gemmini_tiled_matmul_ws_layernorm
  gemmini_tiled_matmul_ws_softmax
  gemmini_tiled_matmul_ws_perf
  gemmini_tiled_matmul_cpu
  gemmini_tiled_matmul_option
  gemmini_transpose
  gemmini_matrix_add
  gemmini_resadd
  gemmini_resadd_stride
  gemmini_global_average
  gemmini_gemmini_counter
  gemmini_template
  COMMENT "Building all Gemmini workloads for Buckyball"
  VERBATIM)
