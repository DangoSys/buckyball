# ===----------------------------------------------------------------------===
# The action thread alone: its tests, programs of
# the thread's virtual environment (bb-tests/workloads/lib/sims/thread),
# built by ../../sims/thread/workloads. A program returns its status in bits
# 31:0 of its completion value (0 passes). regression/<backend> lists the
# tests (workloads-bare).
# ===----------------------------------------------------------------------===
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
set(LIB "${DESIGN_REPO}/bb-tests/workloads/lib")
set(MXINT8MM "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8mm_SOURCE_PATH}/workloads")
set(MXINT8QUANT "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8quant_SOURCE_PATH}/workloads")
set(SOFTMAX "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_softmax_SOURCE_PATH}/workloads")
set(LAYERNORM "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_layernorm_SOURCE_PATH}/workloads")
# The thread and its host, as compile definitions.
set(THREAD_DEFS ${DESIGN_THREAD_DEFS})
list(TRANSFORM THREAD_DEFS PREPEND "-D")
set(OUT "${DESIGN_OUTPUT}/workloads")
file(MAKE_DIRECTORY "${OUT}")

add_custom_command(
  OUTPUT "${OUT}/action-thread-dma_roundtrip-baremetal"
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
    -o "${OUT}/action-thread-dma_roundtrip-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-mxint8mm_small-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8MM}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8mm_small.c"
    -o "${OUT}/action-thread-mxint8mm_small-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8mm_small.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-mxint8mm_bias-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8MM}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8mm_bias.c"
    -o "${OUT}/action-thread-mxint8mm_bias-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8mm_bias.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-mxint8quant_rows-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8QUANT}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quant_rows.c"
    -o "${OUT}/action-thread-mxint8quant_rows-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quant_rows.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-mxint8quant_columns-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXINT8QUANT}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quant_columns.c"
    -o "${OUT}/action-thread-mxint8quant_columns-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quant_columns.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-softmax_exact-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${SOFTMAX}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/softmax_exact.c"
    -o "${OUT}/action-thread-softmax_exact-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/softmax_exact.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/action-thread-layernorm_exact-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${LAYERNORM}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/layernorm_exact.c"
    -o "${OUT}/action-thread-layernorm_exact-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/layernorm_exact.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_target(thread-workloads ALL DEPENDS
  "${OUT}/action-thread-dma_roundtrip-baremetal"
  "${OUT}/action-thread-mxint8mm_small-baremetal"
  "${OUT}/action-thread-mxint8mm_bias-baremetal"
  "${OUT}/action-thread-mxint8quant_rows-baremetal"
  "${OUT}/action-thread-mxint8quant_columns-baremetal"
  "${OUT}/action-thread-softmax_exact-baremetal"
  "${OUT}/action-thread-layernorm_exact-baremetal")
