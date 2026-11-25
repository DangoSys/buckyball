# The workloads of the magia hart simulation: its C API and C tests.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_custom_target(design-ctests-build)
add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
add_subdirectory("${DESIGN_REPO}/examples/prefab/harts/magia/workloads/lib" harts/magia/lib)
add_subdirectory("${DESIGN_REPO}/examples/prefab/harts/magia/workloads/ctests" harts/magia/ctests)
