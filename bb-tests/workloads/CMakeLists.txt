# ===----------------------------------------------------------------------===
# The workloads of one simulation of a design, configured with DESIGN_CMAKE,
# generated from its design.pb by `bbdev stack --build`:
#
#   cmake -G Ninja -S bb-tests/workloads -B <build> -DDESIGN_CMAKE=<design.cmake>
#   ninja -C <build> sync-bin | sync-ctest-bin | sync-mlirtest-bin | sync-soctest-bin
#
# Products are copied to ${DESIGN_OUTPUT}/workloads.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.26)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
project(BuckyballWorkloads C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O2 -UNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -UNDEBUG")
set(WORKLOAD_DIR ${CMAKE_CURRENT_LIST_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)

set(BUCKYBALL_REPO_ROOT ${DESIGN_REPO})
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)
set(BAREMETAL_DIR ${BUCKYBALL_REPO_ROOT}/stack/os/baremetal)
# bb_cores.h: the cores of DESIGN_DEFS.
set(RUNTIME_INCLUDE_DIR ${BUCKYBALL_REPO_ROOT}/stack/runtime/include)

find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_CXX riscv64-unknown-elf-g++ REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)
set(LINUX_CC ${CMAKE_C_COMPILER})
set(LINUX_CXX ${CMAKE_CXX_COMPILER})

# runtime, runtime-baremetal, runtime-linux and engine of every program here:
# built by `bbdev stack --build "--design <design> --device <config>"`.
set(BUCKYBALL_CONTROL_TARGET "core${DESIGN_WORKLOAD_CONTROL_HART}")
# The definitions of every program of the design, as compiler flags.
set(DESIGN_DEFINES ${DESIGN_DEFS})
list(TRANSFORM DESIGN_DEFINES PREPEND "-D")

set(BBSIM_LD ${BAREMETAL_DIR}/bbsim.ld)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker"
)

set(OUTPUT_BIN_DIR "${DESIGN_OUTPUT}/workloads")

set(CTEST_MAX_LINES 100)
set(BBHW_MEM_C ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/dma.c)
set(BUCKYBALL_CTEST_C_FLAGS
    -g
    -fno-common
    -O2
    -static
    -march=rv64gc
    -mcmodel=medany
    -fno-builtin-printf
    -specs=nano.specs
    -specs=nosys.specs
    -nostartfiles
    -Wl,-T,${BBSIM_LD}
)
file(
    GLOB BUCKYBALL_BALL_WORKLOAD_DIRS
    CONFIGURE_DEPENDS
    "${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/*/workloads"
)
file(
    GLOB BUCKYBALL_BALL_ISA_DEPS
    CONFIGURE_DEPENDS
    "${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/*/workloads/isa/*"
)

set(BUDDY_MLIR_DIR ${BUCKYBALL_REPO_ROOT}/stack/thirdparty/buddy-mlir)

set(BUCKYBALL_RISCV_MATTR "+xbuckyball,+m,+D")

add_subdirectory(src)

add_custom_target(
    sync-bin
    DEPENDS sync-native-bin sync-mlirtest-bin sync-soctest-bin
)
add_custom_target(build-all ALL DEPENDS sync-bin)
