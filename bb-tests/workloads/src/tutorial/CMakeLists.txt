project(tutorial C)

#-------------------------------------------------------------------------------
# build linux version workload
#-------------------------------------------------------------------------------
set(LINK_FLAGS "-static -Wl,--no-dynamic-linker")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-g++")  
add_executable(tutorial-linux ${TUTORIAL_WORKLOAD_DIR}/tutorial.c)

add_custom_target(tutorial-linux-build ALL DEPENDS
    tutorial-linux
    COMMENT "Building linux workloads for tutorial"
    VERBATIM)

#-------------------------------------------------------------------------------
# build baremetal version workload
#-------------------------------------------------------------------------------
set(ELF_CC "riscv64-unknown-elf-gcc")  

set(C_FLAGS -std=c11 -g -fno-common -O2 -static 
            -fno-builtin-printf -specs=htif_nano.specs)

add_custom_target(tutorial-baremetal-build ALL
    COMMAND ${ELF_CC} ${C_FLAGS} -o tutorial-baremetal ${TUTORIAL_WORKLOAD_DIR}/tutorial.c
    DEPENDS ${TUTORIAL_WORKLOAD_DIR}/tutorial.c
    COMMENT "Building baremetal workloads for tutorial"
    VERBATIM)

#-------------------------------------------------------------------------------
# build all version workload
#-------------------------------------------------------------------------------
add_custom_target(tutorial-build ALL DEPENDS
    tutorial-linux-build
    tutorial-baremetal-build
    VERBATIM)
