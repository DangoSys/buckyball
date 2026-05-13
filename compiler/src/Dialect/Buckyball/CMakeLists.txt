# Generate dialect and ops from TableGen
add_mlir_dialect(Buckyball buckyball)
add_mlir_doc(Buckyball Buckyball Dialects/ -gen-dialect-doc)

# Generate LLVM IR conversions
set(LLVM_TARGET_DEFINITIONS Buckyball.td)
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBuckyballConversionsIncGen)

# Add subdirectories
add_subdirectory(IR)
add_subdirectory(Transforms)
