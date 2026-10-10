set(BUCKYBALL_CTEST_TARGET ${BUCKYBALL_CORE_WORKLOAD_TARGET})
set(BUCKYBALL_CTEST_GROUP_TARGET core-ctests-build)
add_subdirectory(
  ${BUCKYBALL_REPO_ROOT}/examples/balls/f32add/workloads/ctests
  f32add)

list(APPEND BUCKYBALL_CHIP_COMMON_SOURCES
  ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/tlink.c)
add_buckyball_ctest(tlink_lease.c)
