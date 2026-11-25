# The workloads of the boom hart simulation: its own programs.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_custom_target(design-ctests-build)
set(CTEST_GROUP design-ctests-build)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/cpu" harts/boom/cpu)

add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
