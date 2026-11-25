# ===----------------------------------------------------------------------===
# The matmul thread alone: its tests, programs of
# the thread's virtual environment (bb-tests/workloads/lib/sims/thread),
# built by ../../sims/workloads. A program returns its
# status in bits 31:0 of its completion value (0 passes). regression/<backend>
# lists the tests (workloads-bare).
# ===----------------------------------------------------------------------===
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
set(LIB "${DESIGN_REPO}/bb-tests/workloads/lib")
set(SMATMUL "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_smatmul_SOURCE_PATH}/workloads")
set(TRANSPOSE "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_transpose_SOURCE_PATH}/workloads")
set(INT2FP "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_int2fp_SOURCE_PATH}/workloads")
# The thread and its host, as compile definitions.
set(THREAD_DEFS ${DESIGN_THREAD_DEFS})
list(TRANSFORM THREAD_DEFS PREPEND "-D")
set(OUT "${DESIGN_OUTPUT}/workloads")
file(MAKE_DIRECTORY "${OUT}")

add_custom_command(
  OUTPUT "${OUT}/matmul-thread-dma_roundtrip-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa"  ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c"
    -o "${OUT}/matmul-thread-dma_roundtrip-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/matmul-thread-smatmul_16x16-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${SMATMUL}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/smatmul_16x16.c"
    -o "${OUT}/matmul-thread-smatmul_16x16-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/smatmul_16x16.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/matmul-thread-transpose_i8_16x16-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${TRANSPOSE}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/transpose_i8_16x16.c"
    -o "${OUT}/matmul-thread-transpose_i8_16x16-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/transpose_i8_16x16.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/matmul-thread-int2fp_scale-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${INT2FP}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/int2fp_scale.c"
    -o "${OUT}/matmul-thread-int2fp_scale-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/int2fp_scale.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_target(thread-workloads ALL DEPENDS
  "${OUT}/matmul-thread-dma_roundtrip-baremetal"
  "${OUT}/matmul-thread-smatmul_16x16-baremetal"
  "${OUT}/matmul-thread-transpose_i8_16x16-baremetal"
  "${OUT}/matmul-thread-int2fp_scale-baremetal")
