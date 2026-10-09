add_custom_target(poly-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)
set(BUCKYBALL_CTEST_TARGET attention)
add_subdirectory(ctests)


add_custom_target(chip-ctests-build DEPENDS poly-ctests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building poly workloads"
  VERBATIM)
