add_mlir_translation_library(${BB_CHIP_TARGET_PREFIX}BuddyBuckyballToLLVMIRTranslation
  ${BB_COMPILER_COMMON_DIR}/Target/LLVMIR/Dialect/Buckyball/BuckyballToLLVMIRTranslation.cpp

  DEPENDS
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyballConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyball
)
