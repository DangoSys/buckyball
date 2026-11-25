# The workloads of the piplus action hart simulation: the program of its
# threads and the C tests of their Balls (add_ball_ctests, one per Ball).
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_custom_target(design-ctests-build)
set(CTEST_GROUP design-ctests-build)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../threads/action/workloads" threads/action)
add_ball_ctests()

add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
