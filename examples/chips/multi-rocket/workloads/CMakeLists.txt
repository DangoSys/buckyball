add_custom_target(chip-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET chip-ctests-build)
set(BUCKYBALL_CTEST_TARGET rocket)
add_buckyball_ctest(cpu_arithmetic.c)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building multi-rocket workloads"
  VERBATIM)
