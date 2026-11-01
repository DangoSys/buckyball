add_subdirectory(ctests)
foreach(CPU_TARGET IN LISTS DESIGN_CPU_TARGETS)
  set(BUCKYBALL_CTEST_TARGET "${CPU_TARGET}")
  foreach(BALL_CTEST_DIR IN LISTS DESIGN_PROFILES_${CPU_TARGET}_BALL_CTEST_DIRS)
    cmake_path(GET BALL_CTEST_DIR PARENT_PATH BALL_WORKLOADS)
    cmake_path(GET BALL_WORKLOADS PARENT_PATH BALL_ROOT)
    cmake_path(GET BALL_ROOT FILENAME BALL)
    add_subdirectory(
      "${BUCKYBALL_REPO_ROOT}/${BALL_CTEST_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/balls/${CPU_TARGET}/${BALL}/ctests")
  endforeach()
endforeach()
add_subdirectory(mlir_tests)
