set(DEVICE_CONTROL_EXECUTOR "${DESIGN_HART_BUILD_${DESIGN_PROFILES_main_HART_COMPONENT}}/execute.o")
set(DEVICE_TRANSFER_NAME "${DESIGN_NAME}-main-ctest-device_transfer-multicore-baremetal")
set(DEVICE_TRANSFER_DEFINES ${DESIGN_TARGET_DEFS_main})
list(TRANSFORM DEVICE_TRANSFER_DEFINES PREPEND "-D")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/device_transfer.c")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME}
    COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DEVICE_TRANSFER_DEFINES} ${DESIGN_DEFINES}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-main-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/device_transfer.c
        $<TARGET_FILE:bbsw-hart-baremetal> -lgcc
        -o ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME}
    DEPENDS device_transfer.c ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
        $<TARGET_FILE:bbsw-hart-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_target(device-transfer-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME})
add_dependencies(poly-ctests-build device-transfer-build)

set(COHERENCE_NAME "${DESIGN_NAME}-main-ctest-ant_coherence-multicore-baremetal")
set(COHERENCE_IMAGE "${POLY_ANT_IMAGE_DIR}/${DESIGN_NAME}-attention-ctest-design-ant_coherence-ant.bin")
set(COHERENCE_EMBED "${CMAKE_CURRENT_BINARY_DIR}/ant_coherence_image.S")
set(COHERENCE_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM COHERENCE_DEFINES PREPEND "-D")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/ant_coherence_main.c")
file(GENERATE OUTPUT ${COHERENCE_EMBED} CONTENT
    ".section .rodata.ant,\"a\",@progbits\n.balign 8\n.global ant_image, ant_image_end\nant_image:\n.incbin \"${COHERENCE_IMAGE}\"\nant_image_end:\n.section .note.GNU-stack,\"\",@progbits\n")
add_custom_command(
    OUTPUT ${COHERENCE_NAME}
    COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${COHERENCE_DEFINES}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include -I${POLY_ANT_SOURCE_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ant_coherence_main.c
        ${COHERENCE_EMBED} ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
        $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o ${COHERENCE_NAME}
    DEPENDS ant_coherence_main.c ${COHERENCE_IMAGE} ${COHERENCE_EMBED} ${POLY_BOOT_SOURCE}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_FILE:bbsw-hart-baremetal>
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(ant-coherence-build DEPENDS ${COHERENCE_NAME})
add_dependencies(poly-ctests-build ant-coherence-build)

set(POLY_STREAM_NAME "${DESIGN_NAME}-main-ctest-design-stream")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/stream_main.c")
add_executable(${POLY_STREAM_NAME}-linux stream_main.c)
# bb_isa.h with BUCKYBALL_STREAM: its instructions go into the stream.
target_include_directories(
    ${POLY_STREAM_NAME}-linux
    PRIVATE ${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
)
# The Ball instructions of the job go into the attention thread's stream.
target_compile_definitions(
    ${POLY_STREAM_NAME}-linux
    PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_attention} BUCKYBALL_STREAM
)
target_link_libraries(
    ${POLY_STREAM_NAME}-linux
    PRIVATE bbsw-thread-attention-linux bbsw-device-linux runtime-linux
)
add_dependencies(poly-ctests-build ${POLY_STREAM_NAME}-linux)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program.c")
add_executable(poly-main-ctest-program-linux program.c)
target_link_libraries(poly-main-ctest-program-linux PRIVATE runtime-program-linux)
add_dependencies(poly-ctests-build poly-main-ctest-program-linux)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_transfer.c")
add_executable(poly-main-ctest-program_transfer-linux program_transfer.c)
target_link_libraries(poly-main-ctest-program_transfer-linux PRIVATE poly-runtime-linux)
add_dependencies(poly-ctests-build poly-main-ctest-program_transfer-linux)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/heap_tls.c")
set(HEAP_TLS_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-heap_tls-multicore-baremetal")
set(DEVICE_BOOT "${BUCKYBALL_REPO_ROOT}/stack/runtime/boot")
add_custom_command(
  OUTPUT ${HEAP_TLS_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -fno-builtin
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/heap_tls.c
          -lgcc -o ${HEAP_TLS_ELF}
  DEPENDS heap_tls.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c ${DEVICE_BOOT}/device.ld
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(heap-tls-build DEPENDS ${HEAP_TLS_ELF})
add_dependencies(poly-ctests-build heap-tls-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_baremetal.c")
set(PROGRAM_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program-multicore-baremetal")
add_custom_command(
  OUTPUT ${PROGRAM_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/program_baremetal.c
          $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime>
          -lm -lgcc -o ${PROGRAM_ELF}
  DEPENDS program_baremetal.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld runtime-baremetal runtime
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-baremetal-build DEPENDS ${PROGRAM_ELF})
add_dependencies(poly-ctests-build program-baremetal-build)

add_subdirectory(ctests)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_deferred.c")
set(DEFERRED_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_deferred-multicore-baremetal")
add_custom_command(
  OUTPUT ${DEFERRED_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_deferred.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${DEFERRED_ELF}
  DEPENDS program_deferred.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-deferred-build DEPENDS ${DEFERRED_ELF})
add_dependencies(poly-ctests-build program-deferred-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_joined.c")
set(JOINED_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_joined-multicore-baremetal")
add_custom_command(
  OUTPUT ${JOINED_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_joined.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${JOINED_ELF}
  DEPENDS program_joined.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-joined-build DEPENDS ${JOINED_ELF})
add_dependencies(poly-ctests-build program-joined-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_heap.c")
set(HEAP_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_heap-multicore-baremetal")
add_custom_command(
  OUTPUT ${HEAP_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_heap.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${HEAP_ELF}
  DEPENDS program_heap.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-heap-build DEPENDS ${HEAP_ELF})
add_dependencies(poly-ctests-build program-heap-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_clock.c")
set(CLOCK_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_clock-multicore-baremetal")
add_custom_command(
  OUTPUT ${CLOCK_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_clock.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${CLOCK_ELF}
  DEPENDS program_clock.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-clock-build DEPENDS ${CLOCK_ELF})
add_dependencies(poly-ctests-build program-clock-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_fanout.c")
set(FANOUT_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_fanout-multicore-baremetal")
add_custom_command(
  OUTPUT ${FANOUT_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_fanout.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${FANOUT_ELF}
  DEPENDS program_fanout.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-fanout-build DEPENDS ${FANOUT_ELF})
add_dependencies(poly-ctests-build program-fanout-build)


ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_value_identity.c")
set(VALUE_IDENTITY_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_value_identity-multicore-baremetal")
add_custom_command(
  OUTPUT ${VALUE_IDENTITY_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_value_identity.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${VALUE_IDENTITY_ELF}
  DEPENDS program_value_identity.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-value-identity-build DEPENDS ${VALUE_IDENTITY_ELF})
add_dependencies(poly-ctests-build program-value-identity-build)

get_target_property(DEVICE_DATAFLOW_KERNEL_DIR poly-dataflow-multibank-build BINARY_DIR)
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/device_dataflow.c")
set(DEVICE_DATAFLOW_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-device_dataflow-multicore-baremetal")
add_custom_command(
  OUTPUT ${DEVICE_DATAFLOW_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/examples/designs/${DESIGN_NAME}/harts/poly/workloads/kernel
          -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/device_dataflow.c
          ${DEVICE_DATAFLOW_KERNEL_DIR}/copy_attention.o ${DEVICE_DATAFLOW_KERNEL_DIR}/copy_ffn.o
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${DEVICE_DATAFLOW_ELF}
  DEPENDS ${DEVICE_DATAFLOW_KERNEL_DIR}/copy_attention.o ${DEVICE_DATAFLOW_KERNEL_DIR}/copy_ffn.o device_dataflow.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(device-dataflow-build DEPENDS ${DEVICE_DATAFLOW_ELF})
add_dependencies(poly-ctests-build device-dataflow-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_topology.c")
set(TOPOLOGY_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_topology-multicore-baremetal")
add_custom_command(
  OUTPUT ${TOPOLOGY_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_topology.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${TOPOLOGY_ELF}
  DEPENDS program_topology.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-topology-build DEPENDS ${TOPOLOGY_ELF})
add_dependencies(poly-ctests-build program-topology-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_star.c")
set(STAR_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-program_star-multicore-baremetal")
add_custom_command(
  OUTPUT ${STAR_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          -DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}
          -I${DESIGN_DEVICE_STACK}/runtime
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/program
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${CMAKE_CURRENT_SOURCE_DIR}/program_star.c
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
          $<TARGET_FILE:poly-runtime> $<TARGET_FILE:runtime-baremetal>
          $<TARGET_FILE:runtime> $<TARGET_FILE:kernel>
          -lm -lgcc -o ${STAR_ELF}
  DEPENDS program_star.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c
          ${DEVICE_BOOT}/device.ld poly-runtime runtime-baremetal runtime kernel
          ${DEVICE_CONTROL_EXECUTOR} ${DESIGN_HART_BUILD_poly}/execute.o
          ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(program-star-build DEPENDS ${STAR_ELF})
add_dependencies(poly-ctests-build program-star-build)

set(TRANSPOSE_DMA_NAME "${DESIGN_NAME}-main-ctest-cpu_transpose_dma-multicore-baremetal")
set(TRANSPOSE_DMA_IMAGE "${CMAKE_CURRENT_BINARY_DIR}/cpu_transpose_dma_ant")
set(TRANSPOSE_DMA_EMBED "${CMAKE_CURRENT_BINARY_DIR}/cpu_transpose_dma_image.S")
set(TRANSPOSE_DMA_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM TRANSPOSE_DMA_DEFINES PREPEND "-D")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/cpu_transpose_dma.c")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/cpu_transpose_dma_ant.c")
if(NOT DESIGN_PROFILES_attention_THREAD_MEM_BANK_WIDTH EQUAL 128
   OR DESIGN_PROFILES_attention_THREAD_MEM_BANK_ENTRIES LESS 32)
  message(FATAL_ERROR "cpu_transpose_dma needs 128-bit banks with at least32rows")
endif()
add_custom_command(
  OUTPUT ${TRANSPOSE_DMA_IMAGE}.elf ${TRANSPOSE_DMA_IMAGE}.bin
  COMMAND ${ELF_CC} -O2 -g -DNDEBUG -march=rv64im -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
    -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax
    -Wl,-T,${BUCKYBALL_REPO_ROOT}/stack/kernel/program/code.ld
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    ${TRANSPOSE_DMA_DEFINES} ${BUCKYBALL_REPO_ROOT}/stack/kernel/program/start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_transpose_dma_ant.c -o ${TRANSPOSE_DMA_IMAGE}.elf
  COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text
    ${TRANSPOSE_DMA_IMAGE}.elf ${TRANSPOSE_DMA_IMAGE}.bin
  DEPENDS cpu_transpose_dma_ant.c cpu_transpose_dma.h
    ${BUCKYBALL_REPO_ROOT}/stack/kernel/program/start.S
    ${BUCKYBALL_REPO_ROOT}/stack/kernel/program/code.ld ${BUCKYBALL_BALL_ISA_DEPS}
  COMMAND_EXPAND_LISTS VERBATIM)
file(GENERATE OUTPUT ${TRANSPOSE_DMA_EMBED} CONTENT
  ".section .rodata.ant,\"a\",@progbits\n.balign 8\n.global transpose_ant, transpose_ant_end\ntranspose_ant:\n.incbin \"${TRANSPOSE_DMA_IMAGE}.bin\"\ntranspose_ant_end:\n.section .note.GNU-stack,\"\",@progbits\n")
add_custom_command(
  OUTPUT ${TRANSPOSE_DMA_NAME}
  COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
    $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
    ${DESIGN_DEFINES} ${TRANSPOSE_DMA_DEFINES}
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/cpu_transpose_dma.c
    ${TRANSPOSE_DMA_EMBED} ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
    $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o ${TRANSPOSE_DMA_NAME}
  DEPENDS cpu_transpose_dma.c cpu_transpose_dma.h ${TRANSPOSE_DMA_IMAGE}.bin
    ${TRANSPOSE_DMA_EMBED} ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    $<TARGET_FILE:bbsw-hart-baremetal> ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(cpu-transpose-dma-build DEPENDS ${TRANSPOSE_DMA_NAME})
add_dependencies(poly-ctests-build cpu-transpose-dma-build)
