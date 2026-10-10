add_subdirectory(bbhw)
add_subdirectory(${BUCKYBALL_REPO_ROOT}/stack/runtime runtime EXCLUDE_FROM_ALL)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/CMakeLists.txt")
  add_subdirectory(kernel)
endif()
