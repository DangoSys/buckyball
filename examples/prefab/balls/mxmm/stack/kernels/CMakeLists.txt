# ===----------------------------------------------------------------------===
# MXMM kernels: two RVV kernels built against the thread's Ball ISA, and the
# host code that launches them. Added by the stack of a thread that has this
# Ball, which compiles KERNEL_SOURCES and exports KERNEL_EXPORTS.
# ===----------------------------------------------------------------------===
set(RVV "${DESIGN_REPO}/stack/kernels/rvv")
set(MXMM_OBJECTS)
foreach(SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/norm_window.cpp"
    "${RVV}/norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/packmm.cpp"
    "${RVV}/pack.cpp")
  get_filename_component(NAME "${SOURCE}" NAME_WE)
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND "${RVV_CXX}" --target=riscv64 -march=rv64imf_zve32f_zvl128b -mabi=lp64f
      -mcmodel=medany -ffreestanding -O2 -ffp-contract=off -ffunction-sections -fdata-sections
      "-I${DESIGN_REPO}/stack/kernels" "-I${DESIGN_REPO}/bb-tests/workloads/lib" ${THREAD_DEFS}
      -MD -MF ${NAME}.d -c "${SOURCE}" -o ${NAME}.o
    DEPENDS "${SOURCE}"
    DEPFILE ${NAME}.d
    VERBATIM)
  list(APPEND MXMM_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")
endforeach()
foreach(NAME norm_window packmm)
  add_custom_command(
    OUTPUT ${NAME}.elf
    COMMAND "${RVV_LD}" -m elf64lriscv --gc-sections -e ${NAME} --defsym=kernel_entry=${NAME}
      -T "${RVV}/kernel.ld" ${MXMM_OBJECTS} -o ${NAME}.elf
    DEPENDS ${MXMM_OBJECTS} "${RVV}/kernel.ld"
    VERBATIM)
  add_custom_command(
    OUTPUT ${NAME}.image
    COMMAND "${RVV_OBJCOPY}" -O binary -j .header -j .text -j .rodata ${NAME}.elf ${NAME}.image
    DEPENDS ${NAME}.elf
    VERBATIM)
endforeach()
add_custom_target(mxmm-images DEPENDS
  "${CMAKE_CURRENT_BINARY_DIR}/norm_window.image" "${CMAKE_CURRENT_BINARY_DIR}/packmm.image")

set(KERNEL_SOURCES ${KERNEL_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/pack_dispatch.cpp" PARENT_SCOPE)
set(KERNEL_EMBED_DIRS ${KERNEL_EMBED_DIRS} "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
set(KERNEL_DEPENDS ${KERNEL_DEPENDS} mxmm-images PARENT_SCOPE)
set(KERNEL_EXPORTS ${KERNEL_EXPORTS} _mlir_ciface_rvv_norm_window _mlir_ciface_rvv_packmm PARENT_SCOPE)
