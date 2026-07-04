set(BUCKYBALL_TOY_COMMON_DIR ${TOY_WORKLOAD_COMMON_DIR})
include(${REPO_ROOT}/examples/chips/toy/workloads/cmake/BuckyballWorkloads.cmake)

add_custom_target(base-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET base-ctests-build)

add_buckyball_ctests(
  mvin_mvout_acc_test.c
  mvin_mvout_stride_test.c
  mvin_mvout_test.c
  mvin_mvout_unaligned_test.c
  mvin_unaligned_test.c
)
