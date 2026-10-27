add_buckyball_ctest(scu_uart_rx_echo_test.c)
set(SCU_UART_TEST
    "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-scu_uart_rx_echo_test"
)
add_dependencies(soctests-build ${SCU_UART_TEST}-baremetal-build)
