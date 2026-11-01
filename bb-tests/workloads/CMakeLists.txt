# ===----------------------------------------------------------------------===
# The workloads of one simulation of a design, configured with DESIGN_CMAKE,
# generated from its design.pb, and BBSW_CMAKE, the C API (lib/) built for it:
#
#   cmake -G Ninja -S bb-tests/workloads/lib -B <bbsw> -DDESIGN_CMAKE=<design.cmake>
#   ninja -C <bbsw>
#   cmake -G Ninja -S bb-tests/workloads -B <build> -DDESIGN_CMAKE=<design.cmake>
#         -DBBSW_CMAKE=<bbsw>/bbsw.cmake
#   ninja -C <build> sync-bin | sync-ctest-bin | sync-mlirtest-bin | sync-soctest-bin
#
# The C tests here program the hardware through the C API only. The workload
# entry of the simulation (DESIGN_WORKLOADS) adds the tests of the design and
# of its components, and whatever else it runs.
#
# Products are copied to ${DESIGN_OUTPUT}/workloads.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.26)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
project(BuckyballWorkloads C CXX)
include("${DESIGN_CMAKE}")
# The C API: bbsw-* libraries, see lib/CMakeLists.txt.
include("${BBSW_CMAKE}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O2 -UNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -UNDEBUG")
set(WORKLOAD_DIR ${CMAKE_CURRENT_LIST_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)

set(BUCKYBALL_REPO_ROOT ${DESIGN_REPO})

find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_CXX riscv64-unknown-elf-g++ REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)

set(BUCKYBALL_CONTROL_TARGET "core${DESIGN_WORKLOAD_CONTROL_HART}")
# The definitions of every program of the design, as compiler flags.
set(DESIGN_DEFINES ${DESIGN_DEFS})
list(TRANSFORM DESIGN_DEFINES PREPEND "-D")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker"
)

set(OUTPUT_BIN_DIR "${DESIGN_OUTPUT}/workloads")

set(CTEST_MAX_LINES 100)
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
)
# The headers of the C API and of the design's Balls: C tests depend on them.
file(
    GLOB_RECURSE BUCKYBALL_BBSW_HEADERS
    CONFIGURE_DEPENDS
    "${WORKLOAD_DIR}/lib/*.h"
    "${WORKLOAD_DIR}/lib/bbhw/isa/*.c"
)
set(BUCKYBALL_BALL_ISA_DEPS)
foreach(BALL_WORKLOAD_DIR ${DESIGN_BALL_WORKLOADS})
    file(
        GLOB BALL_ISA_HEADERS
        CONFIGURE_DEPENDS
        "${BALL_WORKLOAD_DIR}/isa/*"
    )
    list(APPEND BUCKYBALL_BALL_ISA_DEPS ${BALL_ISA_HEADERS})
endforeach()

# The workload entry adds whatever else it builds to sync-bin.
add_custom_target(sync-bin DEPENDS sync-native-bin sync-soctest-bin)
add_custom_target(build-all ALL DEPENDS sync-bin)

add_subdirectory(src)
