project(sparse C)

#-------------------------------------------------------------------------------
# build baremetal version workload
#-------------------------------------------------------------------------------
set(ELF_CC "riscv64-unknown-elf-gcc")

set(C_FLAGS -std=c11 -g -fno-common -O2 -static
            -fno-builtin-printf -specs=htif_nano.specs)

add_custom_target(sparse-baremetal-build ALL
    COMMAND ${ELF_CC} ${C_FLAGS} -o sparse-baremetal ${SPARSE_WORKLOAD_DIR}/sparse.c
    DEPENDS ${SPARSE_WORKLOAD_DIR}/sparse.c
    COMMENT "Building baremetal workloads for sparse"
    VERBATIM)

#-------------------------------------------------------------------------------
# build all version workload
#-------------------------------------------------------------------------------
add_custom_target(sparse-build ALL DEPENDS
    sparse-baremetal-build
    VERBATIM)
