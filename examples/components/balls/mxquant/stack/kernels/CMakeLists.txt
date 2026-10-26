# ===----------------------------------------------------------------------===
# MXQuant kernels: the host code that drives the Ball. Added by the stack of a
# thread that has this Ball, which compiles KERNEL_SOURCES and exports
# KERNEL_EXPORTS.
# ===----------------------------------------------------------------------===
set(KERNEL_SOURCES ${KERNEL_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/pack.cpp" PARENT_SCOPE)
set(KERNEL_INCLUDES ${KERNEL_INCLUDES}
  "${DESIGN_REPO}/examples/components/balls/mxquant/workloads" PARENT_SCOPE)
set(KERNEL_EXPORTS ${KERNEL_EXPORTS} mxfp8_quant _mlir_ciface_mxfp8_quant PARENT_SCOPE)
