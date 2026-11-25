# The workloads of the piplus solo hart simulation: the C tests of its
# Balls (add_ball_ctests, one per Ball of its accelerator) and its own program.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_custom_target(design-ctests-build)
set(CTEST_GROUP design-ctests-build)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../workloads" harts/solo)
add_ball_ctests()

add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
