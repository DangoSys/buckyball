add_custom_target(base-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET base-ctests-build)

foreach(BUCKYBALL_CTEST_TARGET IN LISTS DESIGN_CPU_TARGETS)
    set(_BANK_NUM ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_NUM})
    set(_BANK_WIDTH ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH})
    set(_BANK_ENTRIES ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
    set(_MAX_PRIVATE_VBANK ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_FRONTEND_MAX_PRIVATE_VBANK})
    if(_MAX_PRIVATE_VBANK EQUAL 0)
        continue()
    endif()
    math(EXPR _word_remainder "${_BANK_WIDTH} % 32")
    if(
        _BANK_NUM LESS 4
        OR _MAX_PRIVATE_VBANK LESS 2
        OR _BANK_ENTRIES LESS 24
        OR _BANK_WIDTH LESS 64
        OR _BANK_WIDTH GREATER 2048
        OR NOT _word_remainder EQUAL 0
    )
        message(
            FATAL_ERROR
            "base DMA tests require 4 banks for ACC, vbank2, >=24 rows for 2D, and 64..2048-bit rows divisible by 32; ${BUCKYBALL_CTEST_TARGET}: banks=${_BANK_NUM}, privateMax=${_MAX_PRIVATE_VBANK}, lines=${_BANK_ENTRIES}, width=${_BANK_WIDTH}"
        )
    endif()
    if(_MAX_PRIVATE_VBANK GREATER_EQUAL 4)
        add_buckyball_ctest(mset_transfer_test.c)
    endif()
    add_buckyball_ctest(mvin_mvout_group_test.c)
    add_buckyball_ctest(mvin_2d_test.c)
    add_buckyball_ctest(mvin_2d_pixel_stride_test.c)
    add_buckyball_ctest(mvin_mvout_acc_test.c)
    add_buckyball_ctest(mvin_mvout_stride_test.c)
    add_buckyball_ctest(mvin_mvout_test.c)
    add_buckyball_ctest(mvin_mvout_unaligned_test.c)
    add_buckyball_ctest(mvin_unaligned_test.c)
endforeach()
