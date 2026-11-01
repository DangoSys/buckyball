add_subdirectory(control)
add_subdirectory(thread)

# The C tests of the Balls of the design's threads, once per Ball.
set(BALL_CTEST_DIRS)
foreach(PROFILE IN LISTS DESIGN_PROFILES)
    list(APPEND BALL_CTEST_DIRS ${DESIGN_PROFILES_${PROFILE}_BALL_CTEST_DIRS})
endforeach()
list(REMOVE_DUPLICATES BALL_CTEST_DIRS)
foreach(BALL_CTEST_DIR IN LISTS BALL_CTEST_DIRS)
    get_filename_component(BALL_WORKLOADS ${BALL_CTEST_DIR} DIRECTORY)
    get_filename_component(BALL_ROOT ${BALL_WORKLOADS} DIRECTORY)
    get_filename_component(BALL ${BALL_ROOT} NAME)
    add_subdirectory(${BUCKYBALL_REPO_ROOT}/${BALL_CTEST_DIR} balls/${BALL})
endforeach()

add_subdirectory(compiler)
