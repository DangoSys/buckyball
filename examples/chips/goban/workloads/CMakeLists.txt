set(BUCKYBALL_CTEST_TARGET goban)

add_subdirectory(ctests)

add_custom_target(chip-workloads-build ALL
  DEPENDS goban-ctests-build
  COMMENT "Building goban workloads"
  VERBATIM)
