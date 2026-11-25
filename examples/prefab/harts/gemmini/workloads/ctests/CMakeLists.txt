set(GEMMINI_CTESTS
    aligned
    conv
    conv_dw
    conv_first_layer
    conv_rect
    conv_rect_pool
    conv_stride
    conv_trans_input_3120
    conv_trans_input_3120_with_kernel_dilation
    conv_trans_output_1203
    conv_trans_weight_0132
    conv_trans_weight_1203
    conv_with_input_dilation
    conv_with_input_dilation_and_neg_padding
    conv_with_input_dilation_and_rot180
    conv_with_kernel_dilation
    conv_with_pool
    conv_with_rot180
    gemmini_counter
    global_average
    matmul
    matmul_os
    matmul_ws
    matrix_add
    mvin_mvout
    mvin_mvout_acc
    mvin_mvout_acc_full
    mvin_mvout_acc_full_stride
    mvin_mvout_acc_stride
    mvin_mvout_acc_zero_stride
    mvin_mvout_block_stride
    mvin_mvout_stride
    mvin_mvout_zeros
    mvin_scale
    padded
    raw_hazard
    resadd
    resadd_stride
    template
    tiled_matmul_cpu
    tiled_matmul_option
    tiled_matmul_os
    tiled_matmul_ws
    tiled_matmul_ws_At
    tiled_matmul_ws_Bt
    tiled_matmul_ws_full_C
    tiled_matmul_ws_igelu
    tiled_matmul_ws_layernorm
    tiled_matmul_ws_low_D
    tiled_matmul_ws_softmax
    transpose
    transpose_scale
)
foreach(TEST IN LISTS GEMMINI_CTESTS)
    set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-${TEST}")
    set(ELF "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal")
    add_custom_command(
        OUTPUT "${ELF}"
        COMMAND
            ${ELF_CC} ${CTEST_C_FLAGS} -DBAREMETAL=1
            -fno-tree-loop-distribute-patterns
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:gemmini-test-headers,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            "${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.c" -lm -o "${ELF}"
        DEPENDS
            "${TEST}.c"
            ${GEMMINI_TEST_HEADERS}
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${NAME}-baremetal-build DEPENDS "${ELF}")
    add_dependencies(component-ctests-build ${NAME}-baremetal-build)
endforeach()
