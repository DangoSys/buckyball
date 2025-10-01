set(MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/models)
set(GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/gemmini)
set(TOY_DIR ${MODELTEST_WORKLOAD_DIR}/toy)

# ============================================================================
# Parse ARCH variable: ARCH="gemmini,other_arch,..."
# ============================================================================
if (DEFINED ARCH)
  string(REPLACE "," ";" ARCH_LIST ${ARCH})
  foreach(arch_item ${ARCH_LIST})
    string(STRIP ${arch_item} arch_item)
    string(TOUPPER ${arch_item} arch_upper)
    set(ARCH_${arch_upper} ON)
    message(STATUS "Enabled architecture: ${arch_item}")
  endforeach()
endif()

# ============================================================================
# Parse MODEL variable: MODEL="lenet,resnet18,bert,..."
# ============================================================================
if (DEFINED MODEL)
  string(REPLACE "," ";" MODEL_LIST ${MODEL})
  foreach(model_item ${MODEL_LIST})
    string(STRIP ${model_item} model_item)
    string(TOUPPER ${model_item} model_upper)
    set(MODEL_${model_upper} ON)
    message(STATUS "Enabled model: ${model_item}")
  endforeach()
endif()

# ============================================================================
# Architecture dependencies: some architectures need specific models
# ============================================================================

add_subdirectory(models)
add_subdirectory(archs)

add_custom_target(ModelTest DEPENDS
  gemmini-model-build
)
