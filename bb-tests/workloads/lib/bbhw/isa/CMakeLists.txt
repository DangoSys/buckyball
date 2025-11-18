# ISA submodule library

set(ISA_SOURCES
    isa.c
    24_mvin.c
    25_mvout.c
    31_fence.c
    32_mul_warp16.c
    26_bbfp_mul.c
    27_matmul_ws.c
    33_im2col.c
    34_transpose.c
    38_relu.c
    39_bbus_config.c
    40_nnlut.c
    41_snn.c
    42_abft_systolic.c
    43_conv.c
    44_cim.c
    7_flush.c
)

# 1. Linux version
add_library(bbisa-linux STATIC IMPORTED)
set_target_properties(bbisa-linux PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a
    COMMAND riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-isa.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-24_mvin.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-25_mvout.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-31_fence.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-32_mul_warp16.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-26_bbfp_mul.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-27_matmul_ws.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-33_im2col.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-34_transpose.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-38_relu.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/39_bbus_config.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-39_bbus_config.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/40_nnlut.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-40_nnlut.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/41_snn.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-41_snn.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/42_abft_systolic.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-42_abft_systolic.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/43_conv.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-43_conv.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/44_cim.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-44_cim.o
        && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-7_flush.o
        && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a
            linux-isa.o
            linux-24_mvin.o
            linux-25_mvout.o
            linux-31_fence.o
            linux-32_mul_warp16.o
            linux-26_bbfp_mul.o
            linux-27_matmul_ws.o
            linux-33_im2col.o
            linux-34_transpose.o
            linux-38_relu.o
            linux-39_bbus_config.o
            linux-40_nnlut.o
            linux-41_snn.o
            linux-42_abft_systolic.o
            linux-43_conv.o
            linux-44_cim.o
            linux-7_flush.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Linux version of ISA library"
)
add_custom_target(bbisa-linux-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-linux.a)

# 2. Baremetal version
add_library(bbisa-baremetal STATIC IMPORTED)
set_target_properties(bbisa-baremetal PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a
    COMMAND riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-isa.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-24_mvin.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-25_mvout.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-31_fence.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-32_mul_warp16.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-26_bbfp_mul.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-27_matmul_ws.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-33_im2col.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-34_transpose.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-38_relu.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/39_bbus_config.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-39_bbus_config.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/40_nnlut.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-40_nnlut.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/41_snn.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-41_snn.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/42_abft_systolic.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-42_abft_systolic.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/43_conv.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-43_conv.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/44_cim.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-44_cim.o
        && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-7_flush.o
        && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a
            baremetal-isa.o
            baremetal-24_mvin.o
            baremetal-25_mvout.o
            baremetal-31_fence.o
            baremetal-32_mul_warp16.o
            baremetal-26_bbfp_mul.o
            baremetal-27_matmul_ws.o
            baremetal-33_im2col.o
            baremetal-34_transpose.o
            baremetal-38_relu.o
            baremetal-39_bbus_config.o
            baremetal-40_nnlut.o
            baremetal-41_snn.o
            baremetal-42_abft_systolic.o
            baremetal-43_conv.o
            baremetal-44_cim.o
            baremetal-7_flush.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building RISC-V Baremetal version of ISA library"
)
add_custom_target(bbisa-baremetal-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-baremetal.a)

# 3. x86 version
add_library(bbisa-x86 STATIC IMPORTED)
set_target_properties(bbisa-x86 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a
    COMMAND gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/isa.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-isa.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/24_mvin.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-24_mvin.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/25_mvout.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-25_mvout.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/31_fence.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-31_fence.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/32_mul_warp16.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-32_mul_warp16.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/26_bbfp_mul.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-26_bbfp_mul.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/27_matmul_ws.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-27_matmul_ws.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/33_im2col.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-33_im2col.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/34_transpose.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-34_transpose.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-38_relu.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/39_bbus_config.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-39_bbus_config.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/40_nnlut.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-40_nnlut.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/41_snn.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-41_snn.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/42_abft_systolic.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-42_abft_systolic.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/43_conv.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-43_conv.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/44_cim.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-44_cim.o
        && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/7_flush.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-7_flush.o
        && ar rcs ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a
            x86-isa.o
            x86-24_mvin.o
            x86-25_mvout.o
            x86-31_fence.o
            x86-32_mul_warp16.o
            x86-26_bbfp_mul.o
            x86-27_matmul_ws.o
            x86-33_im2col.o
            x86-34_transpose.o
            x86-38_relu.o
            x86-39_bbus_config.o
            x86-40_nnlut.o
            x86-41_snn.o
            x86-42_abft_systolic.o
            x86-43_conv.o
            x86-44_cim.o
            x86-7_flush.o
        && ranlib ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building x86_64 version of ISA library"
)
add_custom_target(bbisa-x86-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libbbisa-x86.a)
