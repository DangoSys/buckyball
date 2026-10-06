add_subdirectory(ctests)
add_custom_target(chip-ctests-build DEPENDS omni-interconnect-build)
add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    mlirtest-workloads-build
    soctests-build
)
