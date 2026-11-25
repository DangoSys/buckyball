add_custom_target(poly-ctests-build)
set(CTEST_GROUP poly-ctests-build)
set(CTEST_PROFILE attention)
# Every hart boots through start.S of the C API.
set(POLY_BOOT_SOURCE "$<TARGET_OBJECTS:bbsw-start-baremetal>")
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/poly/workloads/thread/ctests)
# The harts that run Ant threads: the programs here keep state per such hart.
set(POLY_THREAD_HARTS 0)
foreach(HART IN LISTS DESIGN_DEVICE_HARTS)
    if(NOT "${DESIGN_DEVICE_HARTS_${HART}_THREAD_INDICES}" STREQUAL "")
        math(EXPR POLY_THREAD_HARTS "${POLY_THREAD_HARTS} + 1")
    endif()
endforeach()
list(APPEND DESIGN_DEFS POLY_THREAD_HARTS=${POLY_THREAD_HARTS})
list(APPEND DESIGN_DEFINES -DPOLY_THREAD_HARTS=${POLY_THREAD_HARTS})
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/poly/workloads" harts/poly)
add_subdirectory(soc)
add_subdirectory(runtime)


add_dependencies(design-ctests-build poly-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building poly workloads"
  VERBATIM)
