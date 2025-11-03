set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-gcc")

#-------------------------------------------------------------------------------
# Add subdirectories, each compiles three versions
#-------------------------------------------------------------------------------
add_subdirectory(mem)
add_subdirectory(isa)

#-------------------------------------------------------------------------------
# Combine final library - only responsible for linking submodules
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-gcc")

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

#-------------------------------------------------------------------------------
# build linux version library
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")

#-------------------------------------------------------------------------------
# Generate x86_64 version library (for toy project)
#-------------------------------------------------------------------------------
# 3. x86 version - combine x86 versions of submodules
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract ar x ../mem/libbbmem-x86.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract ar x ../isa/libbbisa-x86.a
  COMMAND ar rcs libbbhw-x86.a temp_extract/*.o
  COMMAND ranlib libbbhw-x86.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract
  DEPENDS bbisa-x86-build bbmem-x86-build
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining x86_64 version of bbhw library"
)

add_custom_target(bbhw-x86 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a)

#-------------------------------------------------------------------------------
# Set baremetal compilation flags
#-------------------------------------------------------------------------------

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
