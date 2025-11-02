

#-------------------------------------------------------------------------------
# Generate executables for different platforms
#-------------------------------------------------------------------------------
# single-core baremetal
# function(add_baremetal_target TARGET_NAME MLIR_FILE)
#   set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal.o")
#   set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal")

#   # Compile xx.mlir to xx-baremetal.o
#   add_custom_command(
#     OUTPUT ${OBJ_FILE}
#     COMMAND ${BUDDY_OPT} ${MLIR_FILE}
#           -lower-buckyball |
#         ${BUDDY_TRANSLATE} --buddy-to-llvmir |
#         ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
#           -mattr=+buddyext,+D -float-abi=hard
#           -relocation-model=pic
#           -o ${OBJ_FILE}
#     DEPENDS ${MLIR_FILE}
#     COMMENT "Compiling ${MLIR_FILE} to object file"
#   )

#   # Link xx-baremetal.o to xx-baremetal
#   add_custom_command(

#     OUTPUT ${EXECUTABLE}
#     COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs
#         ${OBJ_FILE} -o ${EXECUTABLE}
#     DEPENDS ${OBJ_FILE}
#     COMMENT "Linking baremetal executable: ${EXECUTABLE}"
#   )

#   # Create target for xx-baremetal
#   add_custom_target(${TARGET_NAME}_baremetal
#     DEPENDS ${EXECUTABLE}
#   )
# endfunction()

# multi-core baremetal
# function(add_baremetal_target TARGET_NAME MLIR_FILE)
#   set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal.o")
#   set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-baremetal")

#   # Compile xx.mlir to xx-baremetal.o
#   add_custom_command(
#     OUTPUT ${OBJ_FILE}
#     COMMAND ${BUDDY_OPT} ${MLIR_FILE}
#           -lower-buckyball=hartId=3 |
#         ${BUDDY_TRANSLATE} --buddy-to-llvmir |
#         ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
#           -mattr=+buddyext,+D -float-abi=hard
#           -relocation-model=pic
#           -o ${OBJ_FILE}
#     DEPENDS ${MLIR_FILE}
#     COMMENT "Compiling ${MLIR_FILE} to object file"
#   )

#   # Link xx-baremetal.o to xx-baremetal
#   add_custom_command(

#     OUTPUT ${EXECUTABLE}
#     COMMAND ${ELF_CC} -O2 -static -specs=htif_nano.specs
#         ${OBJ_FILE} -o ${EXECUTABLE}
#     DEPENDS ${OBJ_FILE}
#     COMMENT "Linking baremetal executable: ${EXECUTABLE}"
#   )

#   # Create target for xx-baremetal
#   add_custom_target(${TARGET_NAME}_baremetal
#     DEPENDS ${EXECUTABLE}
#   )
# endfunction()

# Generate Linux executables
# function(add_linux_target TARGET_NAME MLIR_FILE)
#   set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-linux.o")
#   set(EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-linux")

#   # Compile xx.mlir to xx-linux.o
#   add_custom_command(
#     OUTPUT ${OBJ_FILE}
#     COMMAND ${BUDDY_OPT} ${MLIR_FILE}
#           -lower-buckyball |
#         ${BUDDY_TRANSLATE} --buddy-to-llvmir |
#         ${BUDDY_LLC} -filetype=obj -mtriple=riscv64
#           -mattr=+buddyext,+D -float-abi=hard
#           -o ${OBJ_FILE}
#     DEPENDS ${MLIR_FILE}
#     COMMENT "Compiling ${MLIR_FILE} to Linux object file"
#   )

#   # Link xx-linux.o to xx-linux
#   add_custom_command(
#     OUTPUT ${EXECUTABLE}
#     COMMAND ${LINUX_CC} -O2 -static ${OBJ_FILE} -o ${EXECUTABLE}
#     DEPENDS ${OBJ_FILE}
#     COMMENT "Linking Linux executable: ${EXECUTABLE}"
#   )

#   # Create target for xx-linux - use target name prefix to avoid conflicts
#   add_custom_target(${TARGET_NAME}_linux
#     DEPENDS ${EXECUTABLE}
#   )
# endfunction()


# Unified build function
# function(add_cross_platform_target TARGET_NAME MLIR_FILE)
#   add_multicore_baremetal_target(${TARGET_NAME} ${MLIR_FILE})
#   add_singlecore_baremetal_target(${TARGET_NAME} ${MLIR_FILE})
#   add_linux_target(${TARGET_NAME} ${MLIR_FILE})

#   add_custom_target(${TARGET_NAME}
#     DEPENDS ${TARGET_NAME}_multicore_baremetal ${TARGET_NAME}_singlecore_baremetal ${TARGET_NAME}_linux
#     COMMENT "Building ${TARGET_NAME} for both baremetal and Linux"
#   )
# endfunction()


#-------------------------------------------------------------------------------
# Build list
#-------------------------------------------------------------------------------
# add_cross_platform_target(bb_mvin_mvout ${OPTEST_TOY_DIR}/bb_mvin_mvout.mlir)
# add_cross_platform_target(bb_dma1 ${OPTEST_TOY_DIR}/bb_dma1.mlir)
# add_cross_platform_target(bb_dma2 ${OPTEST_TOY_DIR}/bb_dma2.mlir)
# add_cross_platform_target(bb_dma3 ${OPTEST_TOY_DIR}/bb_dma3.mlir)
# add_cross_platform_target(bb_mul_warp16 ${OPTEST_TOY_DIR}/bb_mul_warp16.mlir)

# add_custom_target(OpTest-toy ALL DEPENDS
#   bb_mvin_mvout
#   bb_dma1
#   bb_dma2
#   bb_dma3
#   bb_mul_warp16
#   )
