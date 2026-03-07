
#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------

# singlecore baremetal
function(add_singlecore_baremetal_target TARGET_NAME MLIR_FILE)
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_singlecore-baremetal.o")
  set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_singlecore-baremetal")

  # Compile xx.mlir to xx-baremetal.o
  # Tile-level tests need: linalg->tile->buckyball->llvm pipeline
  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${BUDDY_OPT} ${MLIR_FILE}
          -convert-linalg-to-tile
          -convert-tile-to-buckyball
          -expand-strided-metadata
          -convert-linalg-to-loops
          -lower-buckyball
          -finalize-memref-to-llvm
          -convert-arith-to-llvm
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

  # Link (use htif_nano.specs crt0)
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
add_optest_target(tile_matmul ${OPTEST_TILE_DIR}/tile_matmul.mlir)
add_optest_target(tile_transpose ${OPTEST_TILE_DIR}/tile_transpose.mlir)
add_optest_target(tile_conv2d ${OPTEST_TILE_DIR}/tile_conv2d.mlir)

add_custom_target(OpTest-tile ALL DEPENDS
  tile_matmul
  tile_transpose
  tile_conv2d
)
