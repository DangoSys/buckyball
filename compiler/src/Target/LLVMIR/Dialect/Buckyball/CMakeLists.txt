add_mlir_translation_library(BuddyBuckyballToLLVMIRTranslation
  BuckyballToLLVMIRTranslation.cpp

  DEPENDS
  BuddyBuckyballConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyBuckyball
  )
