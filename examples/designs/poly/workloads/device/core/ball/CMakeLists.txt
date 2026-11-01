set(BUCKYBALL_CTEST_TARGET main)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)
add_subdirectory(
  ${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_main_THREAD_BALLDOMAIN_BALLS_f32add_SOURCE_PATH}/workloads/ctests
  f32add)
