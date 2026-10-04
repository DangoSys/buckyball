# Compute tile 1 runs these tests; its first worker profile provides the bank geometry.
set(BUCKYBALL_CTEST_TARGET attention)
set(_params "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h")
foreach(_name BANK_WIDTH BANK_LINES BB_PRIVATE_VBANK_MAX)
  file(STRINGS "${_params}" _line REGEX "^#define ${_name} ")
  string(REGEX REPLACE ".* " "" _${_name} "${_line}")
endforeach()
if(NOT _BANK_WIDTH EQUAL 128 OR _BANK_LINES LESS 4 OR _BB_PRIVATE_VBANK_MAX LESS 1)
  message(FATAL_ERROR "Task DMA requires a private bank and four 128-bit rows; actual width=${_BANK_WIDTH}, lines=${_BANK_LINES}, privateMax=${_BB_PRIVATE_VBANK_MAX}")
endif()
foreach(_name BB_COMPUTE_TILES BB_COMPUTE_CORES)
  file(STRINGS "${BUCKYBALL_ISA_DIR}/chip/topology.h" _line REGEX "^#define ${_name} ")
  string(REGEX REPLACE ".* " "" _${_name} "${_line}")
endforeach()
if(_BB_COMPUTE_TILES LESS 1 OR _BB_COMPUTE_CORES LESS 3)
  message(FATAL_ERROR "Task DMA needs a compute tile with a controller and two workers; got tiles=${_BB_COMPUTE_TILES}, cores=${_BB_COMPUTE_CORES}")
endif()
math(EXPR _worker_count "${_BB_COMPUTE_CORES} - 1")
foreach(_case single workers)
  buckyball_ctest_name(_test ${BUCKYBALL_CTEST_TARGET} chip-task_dma_${_case})
  set(_elf "${_test}-multicore-baremetal")
  set(_main "${CMAKE_CURRENT_SOURCE_DIR}/task_dma_${_case}-main.c")
  set(_workers "${CMAKE_CURRENT_SOURCE_DIR}/task_dma_worker.c")
  if(_case STREQUAL "workers")
    list(APPEND _workers "${CMAKE_CURRENT_SOURCE_DIR}/task_mvo_worker.c")
  endif()
  buckyball_enforce_ctest_line_limit("${_main}")
  foreach(_worker IN LISTS _workers)
    buckyball_enforce_ctest_line_limit("${_worker}")
  endforeach()
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_elf}" "${CMAKE_CURRENT_BINARY_DIR}/${_elf}.expect"
    COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
      -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${CMAKE_CURRENT_SOURCE_DIR}/task.ld
      -I${WORKLOAD_LIB_DIR} -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
      -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore -I${BUCKYBALL_ISA_DIR}/chip
      -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET} -I${CMAKE_CURRENT_SOURCE_DIR}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
      "${CMAKE_CURRENT_SOURCE_DIR}/task_start.S" "${_main}" ${_workers}
      "${WORKLOAD_LIB_DIR}/kernel/opensbi-platform/buckyball/task-worker.S" -lgcc
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/expected.py"
      --nm "${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-nm"
      --elf "${CMAKE_CURRENT_BINARY_DIR}/${_elf}" --case ${_case} --workers ${_worker_count}
      --output "${CMAKE_CURRENT_BINARY_DIR}/${_elf}.expect"
    DEPENDS "${_main}" ${_workers} "${CMAKE_CURRENT_SOURCE_DIR}/task_dma.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/task_start.S" "${CMAKE_CURRENT_SOURCE_DIR}/task.ld"
      "${CMAKE_CURRENT_SOURCE_DIR}/expected.py"
      "${WORKLOAD_LIB_DIR}/kernel/opensbi-platform/buckyball/task-worker.S"
      "${WORKLOAD_LIB_DIR}/bbhw/isa/isa.h" "${WORKLOAD_LIB_DIR}/bbhw/isa/32_mset.c"
      "${WORKLOAD_LIB_DIR}/bbhw/isa/33_mvin.c" "${WORKLOAD_LIB_DIR}/bbhw/isa/16_mvout.c"
      "${WORKLOAD_LIB_DIR}/bbhw/isa/13_mvover.c" "${WORKLOAD_LIB_DIR}/bbhw/isa/14_task.c"
      "${WORKLOAD_LIB_DIR}/bbhw/isa/00_fence.c" "${BUCKYBALL_REPO_ROOT}/stack/runtime/include/dma.h"
      "${_params}" "${BUCKYBALL_ISA_DIR}/chip/topology.h")
  add_custom_target(${_test}-ctest-build
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_elf}" "${CMAKE_CURRENT_BINARY_DIR}/${_elf}.expect")
  add_dependencies(chip-ctests-build ${_test}-ctest-build)
endforeach()

# Task context: workers run their kernels through the controller's satp.
buckyball_ctest_name(_test ${BUCKYBALL_CTEST_TARGET} chip-task_context)
set(_elf "${_test}-multicore-baremetal")
set(_main "${CMAKE_CURRENT_SOURCE_DIR}/task_context-main.c")
buckyball_enforce_ctest_line_limit("${_main}")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
  COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${CMAKE_CURRENT_SOURCE_DIR}/task.ld
    -I${WORKLOAD_LIB_DIR} -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
    -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore -I${BUCKYBALL_ISA_DIR}/chip
    -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET} -I${CMAKE_CURRENT_SOURCE_DIR}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${_elf}"
    "${CMAKE_CURRENT_SOURCE_DIR}/task_start.S" "${_main}"
    "${WORKLOAD_LIB_DIR}/kernel/opensbi-platform/buckyball/task-worker.S" -lgcc
  DEPENDS "${_main}" "${CMAKE_CURRENT_SOURCE_DIR}/task_dma.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/task_start.S" "${CMAKE_CURRENT_SOURCE_DIR}/task.ld"
    "${WORKLOAD_LIB_DIR}/kernel/opensbi-platform/buckyball/task-worker.S"
    "${WORKLOAD_LIB_DIR}/bbhw/isa/isa.h" "${WORKLOAD_LIB_DIR}/bbhw/isa/14_task.c"
    "${_params}" "${BUCKYBALL_ISA_DIR}/chip/topology.h")
add_custom_target(${_test}-ctest-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_elf}")
add_dependencies(chip-ctests-build ${_test}-ctest-build)
