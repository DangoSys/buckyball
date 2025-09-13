set(ELF_CC "riscv64-unknown-elf-g++")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")


#-------------------------------------------------------------------------------
# 生成不同平台的库
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc")

# 生成Linux版本的库
add_library(bbhw-linux
  param/bank.cc
  param/bank.h
  param/dma.h
  mem/spad.cc
  mem/spad.h
)

target_include_directories(bbhw-linux PUBLIC ${WORKLOAD_LIB_DIR})

#-------------------------------------------------------------------------------
# build linux version library
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")

#-------------------------------------------------------------------------------
# 生成x86_64版本的库(用于toy项目)
#-------------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a
  COMMAND /usr/bin/g++ -c -fPIC ${CMAKE_CURRENT_SOURCE_DIR}/param/bank.cc -I${CMAKE_CURRENT_SOURCE_DIR} -o bank.o
  COMMAND /usr/bin/g++ -c -fPIC ${CMAKE_CURRENT_SOURCE_DIR}/param/dma.h -I${CMAKE_CURRENT_SOURCE_DIR} -o dma.o
  COMMAND /usr/bin/g++ -c -fPIC ${CMAKE_CURRENT_SOURCE_DIR}/mem/spad.cc -I${CMAKE_CURRENT_SOURCE_DIR} -o spad.o
  COMMAND ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a bank.o dma.o spad.o
  COMMAND rm -f param.o spad.o
  DEPENDS param/bank.cc mem/spad.cc param/bank.h mem/spad.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Building x86_64 version of bbhw library"
)

add_custom_target(bbhw-x86 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbhw-x86.a)

#-------------------------------------------------------------------------------
# 设置baremetal编译标志
#-------------------------------------------------------------------------------

# 生成baremetal版本的库
add_library(bbhw-baremetal STATIC
  param/bank.cc
  param/bank.h
  param/dma.h
  mem/spad.cc
  mem/spad.h
)

target_compile_options(bbhw-baremetal PRIVATE
  -g -fno-common -O2 -static -march=rv64gc
  -fno-builtin-printf -D__BAREMETAL__
)

target_include_directories(bbhw-baremetal PUBLIC ${WORKLOAD_LIB_DIR})

set_target_properties(bbhw-baremetal PROPERTIES C_COMPILER ${ELF_CC} CXX_COMPILER ${ELF_CC})
