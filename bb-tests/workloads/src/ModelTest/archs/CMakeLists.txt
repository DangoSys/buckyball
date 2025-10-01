set(ARCH_GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/archs/gemmini)
set(ARCH_BUCKYBALL_DIR ${MODELTEST_WORKLOAD_DIR}/archs/buckyball)

if (ARCH_GEMMINI)
  add_subdirectory(gemmini)
endif()

if (ARCH_BUCKYBALL)
  add_subdirectory(buckyball)
endif()
