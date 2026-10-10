get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

add_custom_target(toy-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET toy-ctests-build)
set(BUCKYBALL_CTEST_TARGET toy)

add_subdirectory(ctests)
buckyball_add_ball_workload_subdirs(ctests)

add_custom_target(chip-ctests-build DEPENDS toy-ctests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building toy workloads"
  VERBATIM)
