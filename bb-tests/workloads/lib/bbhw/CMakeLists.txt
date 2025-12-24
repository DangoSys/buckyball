#-------------------------------------------------------------------------------
# Add subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(mem)
add_subdirectory(isa)

#-------------------------------------------------------------------------------
# Combine final library - only responsible for linking submodules
#-------------------------------------------------------------------------------

# 1. Linux version - merge Linux versions of submodules
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-linux.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract_linux
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_linux ar x ../mem/libbbmem-linux.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_linux ar x ../isa/libbbisa-linux.a
  COMMAND ar rcs libbbhw-linux.a temp_extract_linux/*.o
  COMMAND ranlib libbbhw-linux.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract_linux
  DEPENDS bbisa-linux-build bbmem-linux-build
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining RISC-V Linux version of bbhw library"
)

add_custom_target(bbhw-linux ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-linux.a)

# 2. Baremetal version - merge Baremetal versions of submodules
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract_baremetal
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_baremetal ar x ../mem/libbbmem-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_baremetal ar x ../isa/libbbisa-baremetal.a
  COMMAND ar rcs libbbhw-baremetal.a temp_extract_baremetal/*.o
  COMMAND ranlib libbbhw-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract_baremetal
  DEPENDS bbisa-baremetal-build bbmem-baremetal-build
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining RISC-V Baremetal version of bbhw library"
)

add_custom_target(bbhw-baremetal ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-baremetal.a)
