set(ELF_CC "riscv64-unknown-elf-g++")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

#-------------------------------------------------------------------------------
# 添加子目录，各自编译三个版本
#-------------------------------------------------------------------------------
add_subdirectory(mem)
add_subdirectory(isa)

#-------------------------------------------------------------------------------
# 组合最终库 - 只负责链接子模块
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")

# 1. Linux版本 - 合并子模块的Linux版本
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-linux.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract_linux
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_linux ar x ../mem/libbbmem-linux.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_linux ar x ../isa/libbbisa-linux.a
  COMMAND ar rcs libbbhw-linux.a temp_extract_linux/*.o
  COMMAND ranlib libbbhw-linux.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract_linux
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mem/libbbmem-linux.a
          ${CMAKE_CURRENT_BINARY_DIR}/isa/libbbisa-linux.a
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining RISC-V Linux version of bbhw library"
)

add_custom_target(bbhw-linux ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-linux.a)

#-------------------------------------------------------------------------------
# build linux version library
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")

#-------------------------------------------------------------------------------
# 生成x86_64版本的库(用于toy项目)
#-------------------------------------------------------------------------------
# 3. x86版本 - 组合子模块的x86版本
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract ar x ../mem/libbbmem-x86.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract ar x ../isa/libbbisa-x86.a
  COMMAND ar rcs libbbhw-x86.a temp_extract/*.o
  COMMAND ranlib libbbhw-x86.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mem/libbbmem-x86.a
          ${CMAKE_CURRENT_BINARY_DIR}/isa/libbbisa-x86.a
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining x86_64 version of bbhw library"
)

add_custom_target(bbhw-x86 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a)

#-------------------------------------------------------------------------------
# 设置baremetal编译标志
#-------------------------------------------------------------------------------

# 2. Baremetal版本 - 合并子模块的Baremetal版本
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E make_directory temp_extract_baremetal
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_baremetal ar x ../mem/libbbmem-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E chdir temp_extract_baremetal ar x ../isa/libbbisa-baremetal.a
  COMMAND ar rcs libbbhw-baremetal.a temp_extract_baremetal/*.o
  COMMAND ranlib libbbhw-baremetal.a
  COMMAND ${CMAKE_COMMAND} -E remove_directory temp_extract_baremetal
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mem/libbbmem-baremetal.a
          ${CMAKE_CURRENT_BINARY_DIR}/isa/libbbisa-baremetal.a
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Combining RISC-V Baremetal version of bbhw library"
)

add_custom_target(bbhw-baremetal ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-baremetal.a)
