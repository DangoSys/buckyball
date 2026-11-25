# The workloads of the pebble hart simulation: the hart's own C and MLIR tests
# with the C tests of its Balls.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

# The MLIR tests of the hart compile through the harness of the stack.
add_subdirectory(
    "${DESIGN_REPO}/bb-tests/workloads/src/MLIRTest"
    "${CMAKE_BINARY_DIR}/src/MLIRTest"
)
add_dependencies(sync-bin sync-mlirtest-bin)

add_custom_target(pebble-ctests-build)
set(CTEST_GROUP pebble-ctests-build)
set(CTEST_PROFILE pebble)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads" harts/pebble)

add_custom_target(design-ctests-build DEPENDS pebble-ctests-build)
add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
