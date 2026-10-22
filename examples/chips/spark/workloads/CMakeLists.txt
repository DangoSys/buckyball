add_custom_target(spark-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET spark-ctests-build)

buckyball_add_ball_workload_subdirs(ctests)

add_custom_target(chip-ctests-build DEPENDS spark-ctests-build)
add_custom_target(chip-mlirtests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    chip-mlirtests-build
  COMMENT "Building spark workloads"
  VERBATIM)
