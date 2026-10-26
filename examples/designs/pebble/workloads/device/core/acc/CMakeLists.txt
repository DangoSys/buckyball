add_subdirectory(ctests)
foreach(BUCKYBALL_TARGET IN LISTS DESIGN_CPU_TARGETS)
  set(BUCKYBALL_CTEST_TARGET "${BUCKYBALL_TARGET}")
  foreach(BUCKYBALL_BALL_DIR IN LISTS DESIGN_TARGET_BALLS_${BUCKYBALL_TARGET})
    add_subdirectory(
      "${BUCKYBALL_REPO_ROOT}/examples/components/balls/${BUCKYBALL_BALL_DIR}/workloads/ctests"
      "${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_TARGET}/${BUCKYBALL_BALL_DIR}/ctests")
  endforeach()
endforeach()
add_subdirectory(mlir_tests)
