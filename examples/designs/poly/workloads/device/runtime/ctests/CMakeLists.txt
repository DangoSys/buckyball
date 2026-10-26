set(BUCKYBALL_CTEST_TARGET main)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)


list(APPEND BUCKYBALL_DESIGN_COMMON_SOURCES
  ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c)
add_buckyball_ctest(corelink_lease.c)
