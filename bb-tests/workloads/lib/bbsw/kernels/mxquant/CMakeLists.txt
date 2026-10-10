get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
add_library(mxquant_backend STATIC dispatch.cpp)
target_link_libraries(mxquant_backend PRIVATE runtime)
target_compile_options(mxquant_backend PRIVATE -O2)
target_compile_features(mxquant_backend PRIVATE cxx_std_20)
target_include_directories(mxquant_backend PRIVATE
  ${RUNTIME_PARAMS} "${REPO_ROOT}/bb-tests/workloads/lib"
  "${REPO_ROOT}/examples/balls/mxquant/workloads")

add_library(mxquant_backend_ant STATIC EXCLUDE_FROM_ALL dispatch.cpp)
target_link_libraries(mxquant_backend_ant PRIVATE runtime_ant_tasks)
target_compile_options(mxquant_backend_ant PRIVATE -O2)
target_compile_features(mxquant_backend_ant PRIVATE cxx_std_20)
target_include_directories(mxquant_backend_ant PRIVATE
  ${RUNTIME_PARAMS} "${REPO_ROOT}/bb-tests/workloads/lib"
  "${REPO_ROOT}/examples/balls/mxquant/workloads")
