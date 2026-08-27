add_custom_target(poly-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)

add_subdirectory(ctests)
add_subdirectory(mlir_tests)
buckyball_add_ball_workload_subdirs(ctests)

add_custom_target(chip-ctests-build DEPENDS poly-ctests-build)
add_custom_target(chip-mlirtests-build DEPENDS poly-mlir-tests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    chip-mlirtests-build
  COMMENT "Building poly workloads"
  VERBATIM)
