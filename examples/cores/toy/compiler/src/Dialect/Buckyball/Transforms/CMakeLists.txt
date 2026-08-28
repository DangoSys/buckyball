add_mlir_dialect_library(${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
