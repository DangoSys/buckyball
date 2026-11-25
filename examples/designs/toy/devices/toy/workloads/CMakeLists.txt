
add_custom_target(toy-ctests-build)
set(CTEST_GROUP toy-ctests-build)
set(CTEST_PROFILE toy)

add_subdirectory(soc)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/toy/workloads" harts/toy)

add_custom_target(design-ctests-build DEPENDS toy-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building toy workloads"
  VERBATIM)
