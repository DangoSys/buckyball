add_custom_target(design-ctests-build DEPENDS component-ctests-build)
add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
