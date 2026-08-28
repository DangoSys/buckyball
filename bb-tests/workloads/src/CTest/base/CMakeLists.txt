add_custom_target(base-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET base-ctests-build)

foreach(BUCKYBALL_CTEST_TARGET IN LISTS BUCKYBALL_TARGETS)
  add_buckyball_ctests(
    mvin_mvout_acc_test.c
    mvin_mvout_stride_test.c
    mvin_mvout_test.c
    mvin_mvout_unaligned_test.c
    mvin_unaligned_test.c
  )
endforeach()
