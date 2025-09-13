project(sparse C)

#-------------------------------------------------------------------------------
# build baremetal version workload
#-------------------------------------------------------------------------------
set(ELF_CC "riscv64-unknown-elf-gcc")

# use gnu11 instead of c11 to support inline assembly
set(C_FLAGS -std=gnu11 -g -fno-common -O2 -static
      -fno-builtin-printf -specs=htif_nano.specs)

add_custom_target(ctest-sparse-build ALL
  COMMAND ${ELF_CC} ${C_FLAGS} -o sparse-baremetal ${SPARSE_WORKLOAD_DIR}/sparse.c ${SPARSE_WORKLOAD_DIR}/mgen.c
  DEPENDS ${SPARSE_WORKLOAD_DIR}/sparse.c ${SPARSE_WORKLOAD_DIR}/mgen.c
  COMMENT "Building baremetal workloads for sparse"
  VERBATIM)
