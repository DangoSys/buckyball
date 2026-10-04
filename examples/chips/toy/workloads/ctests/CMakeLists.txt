add_buckyball_ctests(
  mmio_minimal_test.c
  scu_uart_rx_echo_test.c
  alexnet_gemmini_tail_test.c
  alexnet_gemmini_stride_test.c
  alexnet_gemmini_round_test.c
)

foreach(_id alexnet_gemmini_tail_test alexnet_gemmini_stride_test alexnet_gemmini_round_test)
  buckyball_ctest_name(_test ${BUCKYBALL_CTEST_TARGET} ${_id})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}-baremetal APPEND
    DEPENDS ${BUCKYBALL_REPO_ROOT}/examples/models/toy/alexnet/execute/conv.c
            ${BUCKYBALL_REPO_ROOT}/examples/models/toy/alexnet/execute/conv.h)
endforeach()
