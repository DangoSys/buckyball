# ===----------------------------------------------------------------------===
# One ELF per GVN instance of a piplus model (a slice, a launch): the
# readback harness with the instance's values embedded, the slice and the
# model's operators, linked like the recipe's baremetal ELF with piplus's
# runtime. `bbdev gvn-bemu --debug` builds it once the recipe's tree has built the
# slices and the driver has written each instance's gvn_*.bin.
#
# Input: DESIGN_CMAKE, the config's generated design variables; MODEL.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.22)
project(piplus-gvn NONE)
include(${DESIGN_CMAKE})

list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE DESIGN_DEFINES)
list(
  APPEND DESIGN_DEFINES
  "-DDEVICE_TRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL"
  "-DDEVICE_TRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL"
)

# The instances: <function>.<index>-launch<l>-iteration<i> of the
# slice <function>.<index>; the operators of every graph (each its own
# _binary_<graph>_operators_image_start).
file(GLOB instances LIST_DIRECTORIES true "${DESIGN_OUTPUT}/gvn/${MODEL}/instances/*")
file(GLOB operators "${DESIGN_OUTPUT}/gvn/${MODEL}/npu/*_operators.o")
set(elfs)
foreach(instance ${instances})
  get_filename_component(name ${instance} NAME)
  string(REGEX REPLACE "-launch.*" "" slice ${name})
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
              ${operators}
              ${DESIGN_HART_BUILD_control}/execute.o ${DESIGN_HART_BUILD_dense}/execute.o ${DESIGN_HART_BUILD_action}/execute.o
              ${DESIGN_THREAD_BUILD_dense}/loader.o ${DESIGN_THREAD_BUILD_action}/loader.o
              ${DESIGN_DEVICE_BUILD}/libpiplus-runtime.a ${DESIGN_DEVICE_BUILD}/runtime/libruntime-program.a
              ${DESIGN_DEVICE_BUILD}/runtime/libruntime-baremetal.a ${DESIGN_DEVICE_BUILD}/runtime/libruntime.a
              ${DESIGN_DEVICE_BUILD}/kernel/libkernel.a
              -lm -lgcc -o ${instance}/slice.elf
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/readback.cpp ${DESIGN_OUTPUT}/gvn/${MODEL}/${slice}/slice.o
            ${instance}/gvn_in.bin ${instance}/gvn_table.bin ${instance}/gvn_out.bin ${instance}/gvn_expect.bin
            ${instance}/gvn_ranges.bin ${instance}/gvn_thread.bin
            ${operators}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  list(APPEND elfs ${instance}/slice.elf)
endforeach()
add_custom_target(elfs ALL DEPENDS ${elfs})
