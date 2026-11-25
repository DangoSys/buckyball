# ===----------------------------------------------------------------------===
# The attention thread alone: its tests, programs of
# the thread's virtual environment (bb-tests/workloads/lib/sims/thread),
# built by examples/prefab/sims/thread/workloads. A program returns its
# status in bits 31:0 of its completion value (0 passes). regression/<backend>
# lists the tests (workloads-bare).
# ===----------------------------------------------------------------------===
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
set(LIB "${DESIGN_REPO}/bb-tests/workloads/lib")
set(TRANSPOSE "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_transpose_SOURCE_PATH}/workloads")
set(INT2FP "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_int2fp_SOURCE_PATH}/workloads")
set(MXMM "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxmm_SOURCE_PATH}/workloads")
set(MXINT8MATMUL "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads")
set(MXINT8QUANTIZE "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads")
set(MXQUANT "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_mxquant_SOURCE_PATH}/workloads")
set(TOINT8 "${DESIGN_REPO}/${DESIGN_THREADS_0_BALLDOMAIN_BALLS_toint8_SOURCE_PATH}/workloads")
# The thread and its host, as compile definitions.
set(THREAD_DEFS ${DESIGN_THREAD_DEFS})
list(TRANSFORM THREAD_DEFS PREPEND "-D")
set(OUT "${DESIGN_OUTPUT}/workloads")
file(MAKE_DIRECTORY "${OUT}")

add_custom_command(
  OUTPUT "${OUT}/attention-thread-dma_roundtrip-baremetal"
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
    -o "${OUT}/attention-thread-dma_roundtrip-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dma_roundtrip.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-transpose_i8_16x16-baremetal"
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
    -o "${OUT}/attention-thread-transpose_i8_16x16-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/transpose_i8_16x16.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-int2fp_scale-baremetal"
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
    -o "${OUT}/attention-thread-int2fp_scale-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/int2fp_scale.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-mxint8matmul_small-baremetal"
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
    -o "${OUT}/attention-thread-mxint8matmul_small-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_small.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-mxmm_mxfp8-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXMM}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxmm_mxfp8.c"
    -o "${OUT}/attention-thread-mxmm_mxfp8-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxmm_mxfp8.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-mxint8quantize_edges-baremetal"
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
    -o "${OUT}/attention-thread-mxint8quantize_edges-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxint8quantize_edges.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-mxquant_values-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${MXQUANT}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/mxquant_values.c"
    -o "${OUT}/attention-thread-mxquant_values-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mxquant_values.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_command(
  OUTPUT "${OUT}/attention-thread-toint8_i32-baremetal"
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0 -mno-relax -nostdlib -nostartfiles -Wl,--no-relax
    "-Wl,-T,${LIB}/sims/thread/code.ld"
    "-Wl,--defsym=CODE_BYTES=${DESIGN_THREADS_0_CPU_ANT_CODE_BYTES}"
    "-Wl,--defsym=TLS_BASE=${DESIGN_THREADS_0_CPU_ANT_TLS_BASE}"
    "-Wl,--defsym=TLS_BYTES=${DESIGN_THREADS_0_CPU_ANT_TLS_BYTES}"
    "-Wl,--defsym=MEMORY_BASE=${DESIGN_THREAD_HOST_MEMORY_BASE}"
    "-I${LIB}" "-I${LIB}/bbhw/isa" "-I${TOINT8}" ${THREAD_DEFS}
    "${LIB}/sims/thread/start.S" "${CMAKE_CURRENT_SOURCE_DIR}/toint8_i32.c"
    -o "${OUT}/attention-thread-toint8_i32-baremetal"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/toint8_i32.c" "${LIB}/sims/thread/start.S"
    "${LIB}/sims/thread/code.ld"
  VERBATIM)

add_custom_target(thread-workloads ALL DEPENDS
  "${OUT}/attention-thread-dma_roundtrip-baremetal"
  "${OUT}/attention-thread-transpose_i8_16x16-baremetal"
  "${OUT}/attention-thread-int2fp_scale-baremetal"
  "${OUT}/attention-thread-mxint8matmul_small-baremetal"
  "${OUT}/attention-thread-mxmm_mxfp8-baremetal"
  "${OUT}/attention-thread-mxint8quantize_edges-baremetal"
  "${OUT}/attention-thread-mxquant_values-baremetal"
  "${OUT}/attention-thread-toint8_i32-baremetal")
