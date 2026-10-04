add_custom_target(base-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET base-ctests-build)

foreach(BUCKYBALL_CTEST_TARGET IN LISTS BUCKYBALL_TARGETS)
  set(_params "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h")
  foreach(_name BANK_NUM BANK_WIDTH BANK_LINES BB_PRIVATE_VBANK_MAX)
    file(STRINGS "${_params}" _line REGEX "^#define ${_name} ")
    string(REGEX REPLACE ".* " "" _${_name} "${_line}")
  endforeach()
  if(_BB_PRIVATE_VBANK_MAX EQUAL 0)
    continue()
  endif()
  math(EXPR _word_remainder "${_BANK_WIDTH} % 32")
  if(_BANK_NUM LESS 4 OR _BB_PRIVATE_VBANK_MAX LESS 2 OR _BANK_LINES LESS 24 OR
     _BANK_WIDTH LESS 64 OR _BANK_WIDTH GREATER 2048 OR NOT _word_remainder EQUAL 0)
    message(FATAL_ERROR
      "base DMA tests require 4 banks for ACC, vbank2, >=24 rows for 2D, and 64..2048-bit rows divisible by 32; ${BUCKYBALL_CTEST_TARGET}: banks=${_BANK_NUM}, privateMax=${_BB_PRIVATE_VBANK_MAX}, lines=${_BANK_LINES}, width=${_BANK_WIDTH}")
  endif()
  add_buckyball_ctests(
    mvin_2d_test.c
    mvin_2d_pixel_stride_test.c
    mvin_mvout_acc_test.c
    mvin_mvout_stride_test.c
    mvin_mvout_test.c
    mvin_mvout_unaligned_test.c
    mvin_unaligned_test.c
  )
endforeach()
