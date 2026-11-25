# The workloads of the reduce hart simulation: the C tests of its Balls,.
include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_custom_target(design-ctests-build)
set(CTEST_GROUP design-ctests-build)
add_ball_ctests()
add_custom_target(design-workloads-build ALL DEPENDS design-ctests-build)
