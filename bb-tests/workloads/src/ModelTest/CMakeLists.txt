set(MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/models)
set(GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/gemmini)
set(TOY_DIR ${MODELTEST_WORKLOAD_DIR}/toy)

if (MODEL_ON)
  add_subdirectory(models)
endif()

if (GEMMINI_ON)
  add_subdirectory(gemmini)
endif()

add_custom_target(ModelTest DEPENDS
    gemmini-model-build
  )
