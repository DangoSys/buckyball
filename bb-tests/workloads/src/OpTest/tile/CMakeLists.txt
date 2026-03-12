
#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------

# Resolve toy runtime paths
set(BBSIM_LD ${CTEST_TOY_WORKLOAD_DIR}/bbsim.ld)
set(CRT0_S   ${CTEST_TOY_WORKLOAD_DIR}/crt0.S)

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

  # Link with toy bbsim runtime (BBSimHarness MMIO)
  add_custom_command(
    OUTPUT ${EXECUTABLE}
    COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mcmodel=medany -nostartfiles
        -specs=nano.specs -specs=nosys.specs
        -Wl,-T,${BBSIM_LD}
        ${CRT0_S} ${OBJ_FILE} -o ${EXECUTABLE}
    DEPENDS ${OBJ_FILE} ${CRT0_S} ${BBSIM_LD}
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
