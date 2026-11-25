add_custom_target(tutorial-ctests-build)
set(CTEST_GROUP tutorial-ctests-build)

# Runs on every hart of each CPU profile.
foreach(CTEST_PROFILE IN LISTS DESIGN_CPU_TARGETS)
    add_ctest(tutorial.c)
endforeach()
