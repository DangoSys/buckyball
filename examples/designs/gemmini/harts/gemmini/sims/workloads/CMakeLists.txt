# The workloads of the gemmini hart simulation: its C API, C and MLIR tests.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_subdirectory("${DESIGN_REPO}/examples/prefab/harts/gemmini/workloads/lib" harts/gemmini/lib)
add_subdirectory("${DESIGN_REPO}/examples/prefab/harts/gemmini/workloads/ctests" harts/gemmini/ctests)
add_custom_target(design-ctests-build DEPENDS component-ctests-build)
add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
add_subdirectory("${DESIGN_REPO}/examples/prefab/harts/gemmini/workloads/mlirtests" harts/gemmini/mlirtests)
