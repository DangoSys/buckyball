set(E2E_MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/e2e)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/CMakeLists.txt")
  add_subdirectory(e2e/models)
endif()
