project(bb-tests)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(BB_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUSTOMEXT_DIR ${BB_TESTS_DIR}/customext)
set(WORKLOADS_DIR ${BB_TESTS_DIR}/workloads)
set(UVBB_DIR ${BB_TESTS_DIR}/uvbb)

add_subdirectory(workloads)
add_subdirectory(customext)
# add_subdirectory(uvbb)
