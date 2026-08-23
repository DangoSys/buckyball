add_custom_target(tutorial-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET tutorial-ctests-build)

# Baremetal single-core only (matches Bazel //bb-tests/.../CTest/tutorial).
add_buckyball_singlecore_ctest(
  ${BUCKYBALL_WORKLOAD_CHIP}_tutorial
  ${CMAKE_CURRENT_SOURCE_DIR}
  tutorial.c)
add_dependencies(tutorial-ctests-build
  ${BUCKYBALL_WORKLOAD_CHIP}_tutorial-singlecore-baremetal-build)
