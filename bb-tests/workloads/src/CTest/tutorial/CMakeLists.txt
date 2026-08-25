add_custom_target(tutorial-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET tutorial-ctests-build)

# Baremetal single-core only (matches Bazel //bb-tests/.../CTest/tutorial).
foreach(BUCKYBALL_CTEST_TARGET IN LISTS BUCKYBALL_TARGETS)
  set(BUCKYBALL_TUTORIAL_TEST
    ${BUCKYBALL_WORKLOAD_CHIP}_${BUCKYBALL_CTEST_TARGET}_tutorial)
  add_buckyball_singlecore_ctest(
    ${BUCKYBALL_TUTORIAL_TEST}
    ${CMAKE_CURRENT_SOURCE_DIR}
    tutorial.c)
  add_dependencies(tutorial-ctests-build
    ${BUCKYBALL_TUTORIAL_TEST}-singlecore-baremetal-build)
endforeach()
