add_custom_target(chip-ctests-build)
add_custom_target(chip-mlirtests-build)

add_subdirectory(ctests)

set(BUCKYBALL_CTEST_TARGET goban)
set(BUCKYBALL_CTEST_GROUP_TARGET chip-ctests-build)
add_subdirectory(
  ${BUCKYBALL_REPO_ROOT}/examples/balls/smatmul/workloads/ctests
  ${CMAKE_CURRENT_BINARY_DIR}/smatmul-ctests)

add_custom_target(chip-workloads-build ALL
  DEPENDS chip-ctests-build
  COMMENT "Building goban workloads"
  VERBATIM)
