add_custom_target(design-ctests-build)
add_subdirectory(soc)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/rocket/workloads" harts/rocket)

add_custom_target(design-workloads-build ALL
  DEPENDS design-ctests-build mlirtest-workloads-build soctests-build)
