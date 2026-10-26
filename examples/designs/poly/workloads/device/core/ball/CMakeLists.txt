set(BUCKYBALL_CTEST_TARGET main)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)
add_subdirectory(
  ${BUCKYBALL_REPO_ROOT}/examples/components/balls/f32add/workloads/ctests
  f32add)
