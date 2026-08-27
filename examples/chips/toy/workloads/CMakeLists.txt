get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

add_custom_target(toy-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET toy-ctests-build)
set(BUCKYBALL_CTEST_TARGET toy)

add_subdirectory(ctests)
add_subdirectory(mlir_tests)
buckyball_add_ball_ctest_subdirs()

add_custom_target(chip-ctests-build DEPENDS toy-ctests-build)
add_custom_target(chip-mlirtests-build DEPENDS toy-mlir-tests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    chip-mlirtests-build
  COMMENT "Building toy workloads"
  VERBATIM)
