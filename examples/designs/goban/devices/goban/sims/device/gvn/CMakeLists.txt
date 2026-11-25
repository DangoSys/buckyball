# ===----------------------------------------------------------------------===
# One ELF per GVN slice of a goban model: the readback harness with the
# slice's values embedded, the slice and the model's operators, linked like
# the recipe's baremetal ELF with goban's runtime. `bbdev gvn-bemu --debug` builds it
# once the recipe's tree has built the slices and the driver has written each
# slice's gvn_*.bin.
#
# Input: DESIGN_CMAKE, the config's generated design variables; MODEL.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(goban-gvn NONE)
include(${DESIGN_CMAKE})

list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)
list(TRANSFORM DESIGN_TARGET_DEFS_matmul PREPEND "-D" OUTPUT_VARIABLE TARGET_DEFINES)

add_custom_command(
  OUTPUT ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/mlir-baremetal.o
  COMMAND ${DESIGN_REPO}/stack/result/bin/riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin
            -I${DESIGN_REPO}/stack/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine
            -c ${DESIGN_REPO}/stack/runtime/services/mlir.cpp -o ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/mlir-baremetal.o
  DEPENDS ${DESIGN_REPO}/stack/runtime/services/mlir.cpp
  VERBATIM
)

# The slices: <function>.<index> of the run.
file(GLOB slices LIST_DIRECTORIES true "${DESIGN_OUTPUT}/gvn/${MODEL}/*.[0-9]*")
set(elfs)
foreach(slice ${slices})
  add_custom_command(
    OUTPUT ${slice}/slice.elf
    BYPRODUCTS ${slice}/npu.o
    COMMAND ${DESIGN_REPO}/stack/result/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax -fno-exceptions -fno-rtti -fno-builtin
              ${DESIGN_DEFINES} -I${DESIGN_REPO}/stack/models/lib -I${DESIGN_REPO}/stack/runtime/scheduler
              -I${DESIGN_REPO}/stack/kernel/include -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa
              --embed-dir=${slice} -c ${CMAKE_CURRENT_SOURCE_DIR}/readback.cpp -o ${slice}/npu.o
    COMMAND ${DESIGN_REPO}/stack/result/bin/riscv64-unknown-elf-gcc -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax
              -static -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_REPO}/stack/runtime/boot/device.ld
              ${DESIGN_DEFINES} ${TARGET_DEFINES} -I${DESIGN_REPO}/stack/runtime/services -I${DESIGN_REPO}/stack/kernel/include
              ${DESIGN_REPO}/stack/runtime/boot/start.S ${DESIGN_REPO}/stack/runtime/boot/syscalls.c
              ${DESIGN_REPO}/stack/kernel/src/location.c ${DESIGN_REPO}/stack/runtime/trace/baremetal.c
              ${slice}/npu.o ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/mlir-baremetal.o ${slice}/slice.o
              ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/operators.embed.o
              ${DESIGN_HART_BUILD_goban}/execute.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
              ${DESIGN_DEVICE_BUILD}/runtime/libruntime-baremetal.a ${DESIGN_DEVICE_BUILD}/runtime/libruntime.a
              ${DESIGN_DEVICE_BUILD}/kernel/libkernel.a
              -lm -lgcc -o ${slice}/slice.elf
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/readback.cpp ${slice}/slice.o
            ${slice}/gvn_in.bin ${slice}/gvn_table.bin ${slice}/gvn_out.bin ${slice}/gvn_expect.bin
            ${slice}/gvn_ranges.bin ${slice}/gvn_thread.bin
            ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/mlir-baremetal.o ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/operators.embed.o
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND elfs ${slice}/slice.elf)
endforeach()
add_custom_target(elfs ALL DEPENDS ${elfs})
