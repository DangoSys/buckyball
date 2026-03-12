project(tutorial C)

#-------------------------------------------------------------------------------
# build linux version workload
#-------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc")

add_executable(tutorial-linux ${TUTORIAL_WORKLOAD_DIR}/tutorial.c)

add_custom_target(tutorial-linux-build ALL DEPENDS tutorial-linux
    COMMENT "Building linux workloads for tutorial"
    VERBATIM)

#-------------------------------------------------------------------------------
# build baremetal version workload (BBSimHarness, same flags as toy)
#-------------------------------------------------------------------------------
set(ELF_CC "riscv64-unknown-elf-gcc")
set(BBSIM_LD ${CTEST_TOY_WORKLOAD_DIR}/bbsim.ld)

set(C_FLAGS -std=c11 -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD} -I${WORKLOAD_LIB_DIR})

add_custom_target(tutorial-baremetal-build ALL
    COMMAND ${ELF_CC} ${C_FLAGS}
        -o tutorial-baremetal
        ${CTEST_TOY_WORKLOAD_DIR}/crt0.S
        ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
        ${TUTORIAL_WORKLOAD_DIR}/tutorial.c
    DEPENDS
        ${TUTORIAL_WORKLOAD_DIR}/tutorial.c
        ${CTEST_TOY_WORKLOAD_DIR}/crt0.S
        ${CTEST_TOY_WORKLOAD_DIR}/buckyball.c
    COMMENT "Building baremetal workloads for tutorial"
    VERBATIM)

#-------------------------------------------------------------------------------
# build all
#-------------------------------------------------------------------------------
add_custom_target(tutorial-build ALL DEPENDS
    tutorial-linux-build
    tutorial-baremetal-build
    VERBATIM)
