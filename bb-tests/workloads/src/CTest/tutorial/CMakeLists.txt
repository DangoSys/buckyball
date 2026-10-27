add_custom_target(tutorial-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET tutorial-ctests-build)

# Runs on the selected bare-metal hart.
foreach(BUCKYBALL_CTEST_TARGET IN LISTS DESIGN_CPU_TARGETS)
    set(BUCKYBALL_TUTORIAL_TEST
        "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-tutorial"
    )
    add_buckyball_baremetal_ctest(
      ${BUCKYBALL_TUTORIAL_TEST}
      ${CMAKE_CURRENT_SOURCE_DIR}
      tutorial.c
    )
    add_dependencies(
        tutorial-ctests-build
        ${BUCKYBALL_TUTORIAL_TEST}-baremetal-build
    )
endforeach()
