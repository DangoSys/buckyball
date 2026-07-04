# Generate dialect and ops from the Toy dialect assembly.
set(LLVM_TARGET_DEFINITIONS ToyBuckyball.td)
mlir_tablegen(Buckyball.h.inc -gen-op-decls)
mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
mlir_tablegen(BuckyballTypes.h.inc -gen-typedef-decls -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
add_mlir_dialect_tablegen_target(${BB_CHIP_TARGET_PREFIX}MLIRBuckyballIncGen)
add_mlir_doc(ToyBuckyball ${BB_CHIP_TARGET_PREFIX}Buckyball Dialects/
  -gen-dialect-doc)

# Generate LLVM IR conversions
set(LLVM_TARGET_DEFINITIONS ToyBuckyball.td)
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(${BB_CHIP_TARGET_PREFIX}BuddyBuckyballConversionsIncGen)

# Add subdirectories
add_subdirectory(IR)
add_subdirectory(Transforms)
