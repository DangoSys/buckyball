set(RISCV_GNU_TOOLCHAIN ${CYDIR}/.conda-env/riscv-tools)
set(ELF_CC "riscv64-unknown-elf-gcc")
set(LINUX_CC "riscv64-unknown-linux-gnu-g++")

set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)

set(BUCKYBALL_OPTEST_DIR ${NPU_WORKLOAD_DIR}/buckyball/OpTest)

set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-g++")  



add_custom_target(buckyball-E2ETest ALL DEPENDS

  )
