# ISA子模块库

set(ISA_SOURCES
    isa.cc
    24_mvin.cc
    25_mvout.cc
    31_fence.cc
    32_mul_warp16.cc
    26_bbfp_mul.cc
    27_matmul_ws.cc
    33_im2col.cc
    34_transpose.cc
    7_flush.cc
)

# 1. Linux版本
add_library(bbisa-linux STATIC IMPORTED)
set_target_properties(bbisa-linux PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a
    COMMAND riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-isa.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-24_mvin.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-25_mvout.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-31_fence.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-32_mul_warp16.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-26_bbfp_mul.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-27_matmul_ws.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-33_im2col.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-34_transpose.o && riscv64-unknown-linux-gnu-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.cc -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-7_flush.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a linux-isa.o linux-24_mvin.o linux-25_mvout.o linux-31_fence.o linux-32_mul_warp16.o linux-26_bbfp_mul.o linux-27_matmul_ws.o linux-33_im2col.o linux-34_transpose.o linux-7_flush.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Linux version of ISA library"
)
add_custom_target(bbisa-linux-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a)

# 2. Baremetal版本
add_library(bbisa-baremetal STATIC IMPORTED)
set_target_properties(bbisa-baremetal PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a
    COMMAND riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-isa.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-24_mvin.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-25_mvout.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-31_fence.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-32_mul_warp16.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-26_bbfp_mul.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-27_matmul_ws.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-33_im2col.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-34_transpose.o && riscv64-unknown-elf-g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.cc -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-7_flush.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a baremetal-isa.o baremetal-24_mvin.o baremetal-25_mvout.o baremetal-31_fence.o baremetal-32_mul_warp16.o baremetal-26_bbfp_mul.o baremetal-27_matmul_ws.o baremetal-33_im2col.o baremetal-34_transpose.o baremetal-7_flush.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Baremetal version of ISA library"
)
add_custom_target(bbisa-baremetal-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a)

# 3. x86版本
add_library(bbisa-x86 STATIC IMPORTED)
set_target_properties(bbisa-x86 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a
    COMMAND g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-isa.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-24_mvin.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-25_mvout.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-31_fence.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-32_mul_warp16.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-26_bbfp_mul.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-27_matmul_ws.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-33_im2col.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-34_transpose.o && g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.cc -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-7_flush.o && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a x86-isa.o x86-24_mvin.o x86-25_mvout.o x86-31_fence.o x86-32_mul_warp16.o x86-26_bbfp_mul.o x86-27_matmul_ws.o x86-33_im2col.o x86-34_transpose.o x86-7_flush.o && ranlib ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building x86_64 version of ISA library"
)
add_custom_target(bbisa-x86-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a)
