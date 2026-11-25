add_custom_target(design-ctests-build)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/goban/workloads" harts/goban)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building goban workloads"
  VERBATIM)
