add_buckyball_ctests(
  difftest_overall_small.c
  difftest_overall_medium.c
  difftest_overall_large.c
  gemmini_os_cisc_atranspose_test.c
  gemmini_os_cisc_basic_test.c
  gemmini_os_cisc_btranspose_test.c
  gemmini_os_cisc_shift_test.c
  gemmini_os_risc_abtranspose_test.c
  gemmini_os_risc_atranspose_test.c
  gemmini_os_risc_basic_test.c
  gemmini_os_risc_btranspose_test.c
  gemmini_os_risc_shift_test.c
  gemmini_ws_cisc_conv_test.c
  gemmini_ws_risc_basic_test.c
  gemmini_ws_risc_btranspose_test.c
  gemmini_ws_risc_shift_test.c
)

foreach(size small medium large)
  buckyball_ctest_name(overall_target ${BUCKYBALL_CTEST_TARGET} difftest_overall_${size})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${overall_target}-baremetal
    APPEND DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemmini_os_risc_basic_test.c)
endforeach()
