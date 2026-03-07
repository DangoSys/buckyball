
#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------

# singlecore baremetal
function(add_singlecore_baremetal_target TARGET_NAME MLIR_FILE)
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_singlecore-baremetal.o")
  set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_singlecore-baremetal")

  # Compile xx.mlir to xx-baremetal.o
  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${BUDDY_OPT} ${MLIR_FILE}
          -expand-strided-metadata
          -convert-linalg-to-loops
          -lower-buckyball
          -convert-scf-to-cf
          -finalize-memref-to-llvm
          -convert-arith-to-llvm
          -convert-cf-to-llvm
          -convert-func-to-llvm
          -reconcile-unrealized-casts |
        ${BUDDY_TRANSLATE} --buddy-to-llvmir |
        ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
          -mattr=+buddyext,+D -float-abi=hard
          -relocation-model=pic
          -o ${OBJ_FILE}
    DEPENDS ${MLIR_FILE}
    COMMENT "Compiling ${MLIR_FILE} to singlecore baremetal object"
  )

  # Link (same as CTest singlecore: no start.S, use htif_nano.specs crt0)
  add_custom_command(
    OUTPUT ${EXECUTABLE}
    COMMAND ${ELF_CC} -O2 -static -specs=${HTIF_DIR}/htif_nano.specs -L${HTIF_DIR}
        ${OBJ_FILE} -o ${EXECUTABLE}
    DEPENDS ${OBJ_FILE}
    COMMENT "Linking singlecore baremetal executable: ${EXECUTABLE}"
  )

  add_custom_target(${TARGET_NAME}_singlecore_baremetal
    DEPENDS ${EXECUTABLE}
  )
endfunction()

# Unified build function
function(add_optest_target TARGET_NAME MLIR_FILE)
  add_singlecore_baremetal_target(${TARGET_NAME} ${MLIR_FILE})

  add_custom_target(${TARGET_NAME}
    DEPENDS ${TARGET_NAME}_singlecore_baremetal
    COMMENT "Building ${TARGET_NAME}"
  )
endfunction()

#-------------------------------------------------------------------------------
# Build list
#-------------------------------------------------------------------------------
add_optest_target(bb_mvin_mvout ${OPTEST_TOY_DIR}/bb_mvin_mvout.mlir)
add_optest_target(bb_dma1 ${OPTEST_TOY_DIR}/bb_dma1.mlir)
add_optest_target(bb_dma2 ${OPTEST_TOY_DIR}/bb_dma2.mlir)
add_optest_target(bb_dma3 ${OPTEST_TOY_DIR}/bb_dma3.mlir)
add_optest_target(bb_mul_warp16 ${OPTEST_TOY_DIR}/bb_mul_warp16.mlir)
add_optest_target(bb_im2col ${OPTEST_TOY_DIR}/bb_im2col.mlir)
add_optest_target(bb_quant_dequant ${OPTEST_TOY_DIR}/bb_quant_dequant.mlir)

add_custom_target(OpTest-toy ALL DEPENDS
  bb_mvin_mvout
  bb_dma1
  bb_dma2
  bb_dma3
  bb_mul_warp16
  bb_im2col
  bb_quant_dequant
)
