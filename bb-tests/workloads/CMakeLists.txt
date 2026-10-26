cmake_minimum_required(VERSION 3.26)
project(BuckyballWorkloads C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(WORKLOAD_DIR ${CMAKE_CURRENT_LIST_DIR})
set(WORKLOAD_SRC_DIR ${WORKLOAD_DIR}/src)

get_filename_component(BUCKYBALL_REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)
set(WORKLOAD_LIB_DIR ${WORKLOAD_DIR}/lib)
set(BAREMETAL_DIR ${BUCKYBALL_REPO_ROOT}/stack/os/baremetal)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_CXX riscv64-unknown-elf-g++ REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)
set(LINUX_CC ${CMAKE_C_COMPILER})
set(LINUX_CXX ${CMAKE_CXX_COMPILER})

set(BUCKYBALL_WORKLOAD_DESIGN "" CACHE STRING "Buckyball design for workloads")
if(BUCKYBALL_WORKLOAD_DESIGN STREQUAL "")
    message(
        FATAL_ERROR
        "BUCKYBALL_WORKLOAD_DESIGN is required. Pass -DBUCKYBALL_WORKLOAD_DESIGN=<design>."
    )
endif()
if(NOT IS_DIRECTORY "${BUCKYBALL_REPO_ROOT}/examples/designs/${BUCKYBALL_WORKLOAD_DESIGN}")
    message(FATAL_ERROR "Buckyball design does not exist: ${BUCKYBALL_WORKLOAD_DESIGN}")
endif()

if(NOT DEFINED BUCKYBALL_DESIGN_PB OR BUCKYBALL_DESIGN_PB STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_DESIGN_PB is required")
endif()

if(NOT EXISTS "${BUCKYBALL_DESIGN_PB}")
    message(FATAL_ERROR "missing Design.pb: ${BUCKYBALL_DESIGN_PB}")
endif()
# Everything this build needs from design.pb: ISA headers and the DESIGN_* lists.
execute_process(
    COMMAND
        "${Python3_EXECUTABLE}" "${BUCKYBALL_REPO_ROOT}/stack/scripts/design.py"
        --design-pb "${BUCKYBALL_DESIGN_PB}" --target "${BUCKYBALL_SIM_TARGET}"
        --out "${CMAKE_BINARY_DIR}/generated"
    COMMAND_ERROR_IS_FATAL ANY
)
set_property(
    DIRECTORY
    APPEND
    PROPERTY
        CMAKE_CONFIGURE_DEPENDS
            "${BUCKYBALL_DESIGN_PB}"
            "${BUCKYBALL_REPO_ROOT}/stack/scripts/design.py"
)
include("${CMAKE_BINARY_DIR}/generated/design.cmake")
# runtime, runtime-baremetal, runtime-linux and engine of every program here:
# built by `bbdev stack --build --device`.
set(BUCKYBALL_ISA_DIR "${CMAKE_BINARY_DIR}/generated/isa")
set(BUCKYBALL_CONTROL_TARGET "core${DESIGN_CONTROL_HART}")

set(BBSIM_LD ${BAREMETAL_DIR}/bbsim.ld)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--no-dynamic-linker"
)

set(OUTPUT_BIN_DIR "${BUCKYBALL_WORKLOAD_OUTPUT}")

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
    "${BUCKYBALL_REPO_ROOT}/examples/components/balls/*/workloads"
)
file(
    GLOB BUCKYBALL_BALL_ISA_DEPS
    CONFIGURE_DEPENDS
    "${BUCKYBALL_REPO_ROOT}/examples/components/balls/*/workloads/isa/*"
)

set(BUDDY_MLIR_DIR ${BUCKYBALL_REPO_ROOT}/stack/thirdparty/buddy-mlir)

set(BUCKYBALL_RISCV_MATTR "+xbuckyball,+m,+D")

add_subdirectory(src)

add_custom_target(
    sync-bin
    DEPENDS sync-native-bin sync-mlirtest-bin sync-soctest-bin
)
add_custom_target(build-all ALL DEPENDS sync-bin)
