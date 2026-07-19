add_mlir_dialect_library(${BB_CHIP_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp
  ${BB_REPO_ROOT}/examples/balls/vector/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/im2col/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/fp2int/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/int2fp/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/relu/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/matrix/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyball
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
