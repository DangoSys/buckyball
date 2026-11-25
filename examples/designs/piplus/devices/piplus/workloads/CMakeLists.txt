add_custom_target(design-ctests-build)
# add_ctest registers each C test here.
set(CTEST_GROUP design-ctests-build)

# The control hart, the threads (through their harts) and the Balls of every
# thread (through their threads).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/control/workloads" harts/control)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../threads/dense/workloads" threads/dense)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../threads/action/workloads" threads/action)
add_ball_ctests()

add_custom_target(design-workloads-build ALL
  DEPENDS design-ctests-build mlirtest-workloads-build soctests-build)
