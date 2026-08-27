set(BUCKYBALL_CTEST_TARGET goban)

add_subdirectory(ctests)

add_custom_target(chip-ctests-build DEPENDS goban-ctests-build)
add_custom_target(chip-mlirtests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS chip-ctests-build
  COMMENT "Building goban workloads"
  VERBATIM)
