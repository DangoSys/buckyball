# ===----------------------------------------------------------------------===
# The ffn5 thread alone: its tests, programs of
# the thread's virtual environment (bb-tests/workloads/lib/sims/thread),
# built by ../../sims/thread/workloads. A program returns its status in bits
# 31:0 of its completion value (0 passes). regression/<backend> lists the
# tests (workloads-bare).
# ===----------------------------------------------------------------------===
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
set(LIB "${DESIGN_REPO}/bb-tests/workloads/lib")
set(MXINT5GEMV "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint5gemv_SOURCE_PATH}/workloads")
set(MXINT8ACT "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8act_SOURCE_PATH}/workloads")
set(MXINT8MATMUL "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads")
set(MXINT8QUANTIZE "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads")
set(TRANSPOSE "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_transpose_SOURCE_PATH}/workloads")
# The thread and its host, as compile definitions.
set(THREAD_DEFS ${DESIGN_THREAD_DEFS})
list(TRANSFORM THREAD_DEFS PREPEND "-D")
set(OUT "${DESIGN_OUTPUT}/workloads")
file(MAKE_DIRECTORY "${OUT}")

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-dma_roundtrip-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c"
    -o "${OUT}/ffn5-thread-dma_roundtrip-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-transpose_i8_16x16-baremetal"
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
    -o "${OUT}/ffn5-thread-transpose_i8_16x16-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/transpose_i8_16x16.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-mxint8quantize_edges-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8QUANTIZE}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quantize_edges.c"
    -o "${OUT}/ffn5-thread-mxint8quantize_edges-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quantize_edges.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-mxint8matmul_small-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8MATMUL}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_small.c"
    -o "${OUT}/ffn5-thread-mxint8matmul_small-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_small.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-mxint8act_relu-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8ACT}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8act_relu.c"
    -o "${OUT}/ffn5-thread-mxint8act_relu-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8act_relu.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/ffn5-thread-mxint5gemv_block-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT5GEMV}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint5gemv_block.c"
    -o "${OUT}/ffn5-thread-mxint5gemv_block-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint5gemv_block.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_target(thread-workloads ALL DEPENDS
  "${OUT}/ffn5-thread-dma_roundtrip-baremetal"
  "${OUT}/ffn5-thread-transpose_i8_16x16-baremetal"
  "${OUT}/ffn5-thread-mxint8quantize_edges-baremetal"
  "${OUT}/ffn5-thread-mxint8matmul_small-baremetal"
  "${OUT}/ffn5-thread-mxint8act_relu-baremetal"
  "${OUT}/ffn5-thread-mxint5gemv_block-baremetal")
