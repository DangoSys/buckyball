# MEM子模块库

set(MEM_SOURCES
    bank.cc
    spad.cc
)

# 1. Linux版本
add_library(bbmem-linux STATIC IMPORTED)
set_target_properties(bbmem-linux PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a
    COMMAND riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/bank.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-bank.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/spad.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-spad.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a linux-bank.o linux-spad.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Linux version of MEM library"
)
add_custom_target(bbmem-linux-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-linux.a)

# 2. Baremetal版本
add_library(bbmem-baremetal STATIC IMPORTED)
set_target_properties(bbmem-baremetal PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a
    COMMAND riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/bank.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-bank.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/spad.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-spad.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a baremetal-bank.o baremetal-spad.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Baremetal version of MEM library"
)
add_custom_target(bbmem-baremetal-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-baremetal.a)

# 3. x86版本
add_library(bbmem-x86 STATIC IMPORTED)
set_target_properties(bbmem-x86 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-x86.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-x86.a
    COMMAND g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/bank.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-bank.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/spad.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-spad.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-x86.a x86-bank.o x86-spad.o && ranlib ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-x86.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building x86_64 version of MEM library"
)
add_custom_target(bbmem-x86-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbmem-x86.a)
