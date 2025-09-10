set(MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/models)
set(GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/gemmini)
set(TOY_DIR ${MODELTEST_WORKLOAD_DIR}/toy)

add_subdirectory(models)
add_subdirectory(gemmini)


add_custom_target(ModelTest DEPENDS
    gemmini-model-build

  )
