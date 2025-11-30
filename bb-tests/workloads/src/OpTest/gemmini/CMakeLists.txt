# Toolchain resolvers (prefer RISCV_GNU_TOOLCHAIN, fallback to PATH names)
if(DEFINED ENV{RISCV_GNU_TOOLCHAIN})
  set(ELF_CC "$ENV{RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc")
  set(LINUX_CC "$ENV{RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc")
else()
  set(ELF_CC "riscv64-unknown-elf-gcc")
  set(LINUX_CC "riscv64-unknown-linux-gnu-gcc")
endif()




#-------------------------------------------------------------------------------
# Define common compilation step functions .mlir -> .o
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------
# Generate baremetal executables
function(add_baremetal_target TARGET_NAME MLIR_FILE)
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal.o")
  set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal")

  # Compile xx.mlir to xx-baremetal.o
  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${BUDDY_OPT} ${MLIR_FILE}
          -convert-linalg-to-gemmini
          -expand-strided-metadata
          -convert-linalg-to-loops
          -lower-gemmini |
        ${BUDDY_TRANSLATE} --buddy-to-llvmir |
        ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
          -mattr=+buddyext,+D -float-abi=hard
          -relocation-model=pic
          -o ${OBJ_FILE}
    DEPENDS ${MLIR_FILE}
    COMMENT "Compiling ${MLIR_FILE} to object file"
  )

  # Link xx-baremetal.o to xx-baremetal
  add_custom_command(

    OUTPUT ${EXECUTABLE}
    COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs
        ${OBJ_FILE} -o ${EXECUTABLE}
    DEPENDS ${OBJ_FILE}
    COMMENT "Linking baremetal executable: ${EXECUTABLE}"
  )

  # Create target for xx-baremetal
  add_custom_target(${TARGET_NAME}_baremetal
    DEPENDS ${EXECUTABLE}
  )
endfunction()

# Generate Linux executables
function(add_linux_target TARGET_NAME MLIR_FILE)
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-linux.o")
  set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-linux")

  # Compile xx.mlir to xx-linux.o
  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${BUDDY_OPT} ${MLIR_FILE}
          -convert-linalg-to-gemmini
          -expand-strided-metadata
          -convert-linalg-to-loops
          -lower-gemmini |
        ${BUDDY_TRANSLATE} --buddy-to-llvmir |
        ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
          -mattr=+buddyext,+D -float-abi=hard
          -o ${OBJ_FILE}
    DEPENDS ${MLIR_FILE}
    COMMENT "Compiling ${MLIR_FILE} to Linux object file"
  )

  # Link xx-linux.o to xx-linux
  add_custom_command(
    OUTPUT ${EXECUTABLE}
    COMMAND ${LINUX_CC} -O2 -static ${OBJ_FILE} -o ${EXECUTABLE}
    DEPENDS ${OBJ_FILE}
    COMMENT "Linking Linux executable: ${EXECUTABLE}"
  )

  # Create target for xx-linux - use target name prefix to avoid conflicts
  add_custom_target(${TARGET_NAME}_linux
    DEPENDS ${EXECUTABLE}
  )
endfunction()

# Create executables for different platforms
function(add_cross_platform_target TARGET_NAME MLIR_FILE)
  add_baremetal_target(${TARGET_NAME} ${MLIR_FILE})
  add_linux_target(${TARGET_NAME} ${MLIR_FILE})

  # Create a master target that builds both platforms simultaneously
  add_custom_target(${TARGET_NAME}
    DEPENDS ${TARGET_NAME}_baremetal ${TARGET_NAME}_linux
    COMMENT "Building ${TARGET_NAME} for both baremetal and Linux"
  )
endfunction()

#-------------------------------------------------------------------------------
# Build list
#-------------------------------------------------------------------------------
add_cross_platform_target(matrix-add ${OPTEST_GEMMINI_DIR}/matrix-add.mlir)
add_cross_platform_target(matrix-add-scale ${OPTEST_GEMMINI_DIR}/matrix-add-scale.mlir)
add_cross_platform_target(mvin-mvout ${OPTEST_GEMMINI_DIR}/mvin-mvout.mlir)
add_cross_platform_target(transpose ${OPTEST_GEMMINI_DIR}/transpose.mlir)
add_cross_platform_target(compute-accumulated ${OPTEST_GEMMINI_DIR}/compute-accumulated.mlir)

add_cross_platform_target(matmul ${OPTEST_GEMMINI_DIR}/matmul.mlir)
add_cross_platform_target(matmul-os ${OPTEST_GEMMINI_DIR}/matmul-os.mlir)
add_cross_platform_target(matmul-ws ${OPTEST_GEMMINI_DIR}/matmul-ws.mlir)
add_cross_platform_target(batch_matmul ${OPTEST_GEMMINI_DIR}/batch_matmul.mlir)

add_cross_platform_target(tile-matmul ${OPTEST_GEMMINI_DIR}/tile-matmul.mlir)
add_cross_platform_target(tile-matmul-os ${OPTEST_GEMMINI_DIR}/tile-matmul-os.mlir)
add_cross_platform_target(tile-matmul-ws-relu ${OPTEST_GEMMINI_DIR}/tile-matmul-ws-relu.mlir)
add_cross_platform_target(tile-matmul-ws-igelu ${OPTEST_GEMMINI_DIR}/tile-matmul-ws-igelu.mlir)
add_cross_platform_target(tile-matmul-ws-softmax ${OPTEST_GEMMINI_DIR}/tile-matmul-ws-softmax.mlir)
add_cross_platform_target(tile-matmul-ws-layernorm ${OPTEST_GEMMINI_DIR}/tile-matmul-ws-layernorm.mlir)

add_cross_platform_target(conv_2d_nchw_fchw_f32 ${OPTEST_GEMMINI_DIR}/conv_2d_nchw_fchw_f32.mlir)
add_cross_platform_target(conv_2d_nchw_fchw_i8 ${OPTEST_GEMMINI_DIR}/conv_2d_nchw_fchw_i8.mlir)
add_cross_platform_target(conv_2d_nhwc_hwcf_i8 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_hwcf_i8.mlir)
add_cross_platform_target(conv_2d_nhwc_hwcf_f32 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_hwcf_f32.mlir)
add_cross_platform_target(conv_2d_nhwc_hwcf_5x5_i8 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_hwcf_5x5_i8.mlir)
add_cross_platform_target(conv_2d_nhwc_fhwc_f32 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_fhwc_f32.mlir)
add_cross_platform_target(conv_2d_nhwc_fhwc_i8 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_fhwc_i8.mlir)
add_cross_platform_target(conv_2d_nhwc_fhwc_5x5_i8 ${OPTEST_GEMMINI_DIR}/conv_2d_nhwc_fhwc_5x5_i8.mlir)

add_cross_platform_target(tile-conv ${OPTEST_GEMMINI_DIR}/tile-conv.mlir)
add_cross_platform_target(tile-conv-relu ${OPTEST_GEMMINI_DIR}/tile-conv-relu.mlir)
add_cross_platform_target(tile-conv-softmax ${OPTEST_GEMMINI_DIR}/tile-conv-softmax.mlir)
add_cross_platform_target(tile-rect-conv ${OPTEST_GEMMINI_DIR}/tile-rect-conv.mlir)
add_cross_platform_target(tile-conv-igelu ${OPTEST_GEMMINI_DIR}/tile-conv-igelu.mlir)
add_cross_platform_target(tile-conv-layernorm ${OPTEST_GEMMINI_DIR}/tile-conv-layernorm.mlir)

add_custom_target(OpTest-gemmini ALL DEPENDS
  matrix-add
  matrix-add-scale
  mvin-mvout
  transpose
  compute-accumulated
  matmul
  matmul-os
  matmul-ws
  batch_matmul
  tile-matmul
  tile-matmul-os
  tile-matmul-ws-relu
  tile-matmul-ws-igelu
  tile-matmul-ws-softmax
  tile-matmul-ws-layernorm
  conv_2d_nchw_fchw_f32
  conv_2d_nchw_fchw_i8
  conv_2d_nhwc_hwcf_i8
  conv_2d_nhwc_hwcf_f32
  conv_2d_nhwc_hwcf_5x5_i8
  conv_2d_nhwc_fhwc_f32
  conv_2d_nhwc_fhwc_i8
  conv_2d_nhwc_fhwc_5x5_i8
  tile-conv
  tile-conv-relu
  tile-conv-softmax
  tile-rect-conv
  tile-conv-igelu
  tile-conv-layernorm
)
