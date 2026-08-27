add_custom_target(chip-ctests-build)
add_custom_target(chip-mlirtests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS chip-ctests-build chip-mlirtests-build
  COMMENT "multi-rocket has no chip-local workloads"
  VERBATIM)
