add_custom_target(poly-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)
set(BUCKYBALL_CTEST_TARGET attention)
set(POLY_BOOT_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device/common/controllers_start.S)
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/device/core/thread/ctests)
add_subdirectory(device)


add_custom_target(design-ctests-build DEPENDS poly-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building poly workloads"
  VERBATIM)
