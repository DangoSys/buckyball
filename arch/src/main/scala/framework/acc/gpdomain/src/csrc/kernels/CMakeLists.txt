# ===----------------------------------------------------------------------===
# RVV kernels the GP domain's RTL tests run, built by build.rs:
#
#   cmake -S kernels -B <out> -DREPO=<repo root>
#
# Products, in the build directory:
#   <kernel>.elf    the linked kernel, loaded by the reference model
#   <kernel>.image  what mvin_kernel loads
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(gpdomain-test-kernels NONE)

find_program(RVV_CXX clang++ PATHS "${REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
find_program(RVV_LD riscv64-unknown-elf-ld PATHS "${REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
find_program(RVV_OBJCOPY riscv64-unknown-elf-objcopy PATHS "${REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
set(RVV_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")

# ===----------------------------------------------------------------------===
# Objects. Every kernel links all of them and keeps what its entry reaches.
# ===----------------------------------------------------------------------===
set(RVV_OBJECTS)
foreach(
  SOURCE
  silu
  snake
  quant
  math/exp
  math/sin
)
  get_filename_component(NAME "${SOURCE}" NAME)
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND "${RVV_CXX}"
              --target=riscv64 -march=rv64imf_zve32f_zvl128b -mabi=lp64f -mcmodel=medany -ffreestanding -O2 -ffp-contract=off -ffunction-sections -fdata-sections -MD -MF ${NAME}.d -c "${RVV_SOURCE}/${SOURCE}.cpp" -o ${NAME}.o
    DEPENDS "${RVV_SOURCE}/${SOURCE}.cpp"
    DEPFILE ${NAME}.d
    VERBATIM
  )
  list(APPEND RVV_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")
endforeach()

# ===----------------------------------------------------------------------===
# Images: kernel name and entry symbol.
# ===----------------------------------------------------------------------===
set(RVV_IMAGES)
foreach(
  KERNEL
  silu:rvv_silu_launch
  swiglu:rvv_swiglu_launch
  snake:rvv_snake
  quant:rvv_quant
)
  string(REPLACE ":" ";" KERNEL "${KERNEL}")
  list(GET KERNEL 0 NAME)
  list(GET KERNEL 1 ENTRY)
  add_custom_command(
    OUTPUT ${NAME}.elf
    COMMAND "${RVV_LD}" -m elf64lriscv
              --gc-sections -e ${ENTRY}
              --defsym=kernel_entry=${ENTRY} -T "${RVV_SOURCE}/kernel.ld" ${RVV_OBJECTS} -o ${NAME}.elf
    DEPENDS ${RVV_OBJECTS}
            "${RVV_SOURCE}/kernel.ld"
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${NAME}.image
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata ${NAME}.elf ${NAME}.image
    DEPENDS ${NAME}.elf
    VERBATIM
  )
  list(APPEND RVV_IMAGES "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.image")
endforeach()
add_custom_target(rvv-images ALL DEPENDS ${RVV_IMAGES})
