# MEM submodule library

file(GLOB MEM_C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

# 1. Linux version
add_library(bbmem-linux STATIC IMPORTED)
set_target_properties(bbmem-linux PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a)

set(LINUX_OBJS "")
foreach(src ${MEM_C_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/linux-${name}.o")
  list(APPEND LINUX_OBJS ${obj})
  add_custom_command(
    OUTPUT ${obj}
    COMMAND riscv64-unknown-linux-gnu-gcc -c ${src} -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/../isa -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o ${obj}
    DEPENDS ${src}
    COMMENT "Compiling ${name}.c for Linux"
  )
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a
    COMMAND riscv64-unknown-linux-gnu-ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a ${LINUX_OBJS}
    DEPENDS ${LINUX_OBJS}
    COMMENT "Building RISC-V Linux version of MEM library"
)
add_custom_target(bbmem-linux-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a)

# 2. Baremetal version
add_library(bbmem-baremetal STATIC IMPORTED)
set_target_properties(bbmem-baremetal PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a)

set(BAREMETAL_OBJS "")
foreach(src ${MEM_C_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/baremetal-${name}.o")
  list(APPEND BAREMETAL_OBJS ${obj})
  add_custom_command(
    OUTPUT ${obj}
    COMMAND riscv64-unknown-elf-gcc -c ${src} -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/../isa -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o ${obj}
    DEPENDS ${src}
    COMMENT "Compiling ${name}.c for Baremetal"
  )
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a
    COMMAND riscv64-unknown-elf-ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a ${BAREMETAL_OBJS}
    DEPENDS ${BAREMETAL_OBJS}
    COMMENT "Building RISC-V Baremetal version of MEM library"
)
add_custom_target(bbmem-baremetal-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a)
