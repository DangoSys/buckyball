set(LLVM_TARGET_DEFINITIONS PebbleBuckyball.td)
mlir_tablegen(Buckyball.h.inc -gen-op-decls)
mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
mlir_tablegen(BuckyballTypes.h.inc -gen-typedef-decls -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
add_public_tablegen_target(${BB_CHIP_TARGET_PREFIX}MLIRBuckyballIncGen)
add_mlir_doc(PebbleBuckyball ${BB_CHIP_TARGET_PREFIX}Buckyball Dialects/
  -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS PebbleBuckyball.td)
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(${BB_CHIP_TARGET_PREFIX}BuddyBuckyballConversionsIncGen)

add_subdirectory(IR)
add_subdirectory(Transforms)
