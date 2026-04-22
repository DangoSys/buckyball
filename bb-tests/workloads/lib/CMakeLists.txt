add_subdirectory(bbhw)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/CMakeLists.txt")
  add_subdirectory(kernel)
endif()
