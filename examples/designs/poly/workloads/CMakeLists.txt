add_custom_target(poly-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET poly-ctests-build)
set(BUCKYBALL_CTEST_TARGET attention)
set(POLY_BOOT_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device/common/controllers_start.S)
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/device/core/thread/ctests)
# The cores that run Ant threads: the programs here keep state per such core.
set(POLY_THREAD_CORES 0)
foreach(CORE IN LISTS DESIGN_DEVICE_CORES)
    if(NOT "${DESIGN_DEVICE_CORES_${CORE}_THREAD_INDICES}" STREQUAL "")
        math(EXPR POLY_THREAD_CORES "${POLY_THREAD_CORES} + 1")
    endif()
endforeach()
list(APPEND DESIGN_DEFS POLY_THREAD_CORES=${POLY_THREAD_CORES})
list(APPEND DESIGN_DEFINES -DPOLY_THREAD_CORES=${POLY_THREAD_CORES})
add_subdirectory(device)


add_custom_target(design-ctests-build DEPENDS poly-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building poly workloads"
  VERBATIM)
