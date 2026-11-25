# ===----------------------------------------------------------------------===
# One ELF per GVN instance of a poly model (a slice, a step, a launch): the
# readback harness with the instance's values embedded, the slice and the
# model's operators, linked like the recipe's baremetal ELF with poly's
# runtime. `bbdev gvn-bemu --debug` builds it once the recipe's tree has built the
# slices and the driver has written each instance's gvn_*.bin.
#
# Input: DESIGN_CMAKE, the config's generated design variables; MODEL.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(poly-gvn NONE)
include(${DESIGN_CMAKE})

list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)

# The instances: <function>.<index>-step<s>-launch<l>-iteration<i> of the
# slice <function>.<index>.
file(GLOB instances LIST_DIRECTORIES true "${DESIGN_OUTPUT}/gvn/${MODEL}/instances/*")
set(elfs)
foreach(instance ${instances})
  get_filename_component(name ${instance} NAME)
  string(REGEX REPLACE "-step.*" "" slice ${name})
  add_custom_command(
    OUTPUT ${instance}/slice.elf
    BYPRODUCTS ${instance}/npu.o
    COMMAND ${DESIGN_REPO}/stack/result/bin/riscv64-unknown-elf-g++ -std=c++20 -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -fno-exceptions -fno-rtti
              ${DESIGN_DEFINES} -I${DESIGN_REPO}/stack/models/lib -I${DESIGN_REPO}/stack/runtime/scheduler
              -I${DESIGN_REPO}/stack/kernel/include -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa
              --embed-dir=${instance} -c ${CMAKE_CURRENT_SOURCE_DIR}/readback.cpp -o ${instance}/npu.o
    COMMAND ${DESIGN_REPO}/stack/result/bin/riscv64-unknown-elf-gcc -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany
              -static -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${DESIGN_REPO}/stack/runtime/boot/device.ld
              ${DESIGN_DEFINES} -I${DESIGN_REPO}/stack/kernel/include
              ${DESIGN_REPO}/stack/runtime/boot/start.S ${DESIGN_REPO}/stack/runtime/boot/syscalls.c
              ${DESIGN_REPO}/stack/runtime/trace/baremetal.c
              ${instance}/npu.o ${DESIGN_OUTPUT}/gvn/${MODEL}/${slice}/slice.o
              ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/operators.embed.o
              ${DESIGN_HART_BUILD_poly}/execute.o
              ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
              ${DESIGN_DEVICE_BUILD}/libpoly-runtime.a ${DESIGN_DEVICE_BUILD}/runtime/libruntime-program.a
              ${DESIGN_DEVICE_BUILD}/runtime/libruntime-baremetal.a ${DESIGN_DEVICE_BUILD}/runtime/libruntime.a
              ${DESIGN_DEVICE_BUILD}/kernel/libkernel.a
              -lm -lgcc -o ${instance}/slice.elf
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/readback.cpp ${DESIGN_OUTPUT}/gvn/${MODEL}/${slice}/slice.o
            ${instance}/gvn_in.bin ${instance}/gvn_table.bin ${instance}/gvn_out.bin ${instance}/gvn_expect.bin
            ${instance}/gvn_ranges.bin ${instance}/gvn_thread.bin
            ${DESIGN_OUTPUT}/gvn/${MODEL}/npu/operators.embed.o
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND elfs ${instance}/slice.elf)
endforeach()
add_custom_target(elfs ALL DEPENDS ${elfs})
