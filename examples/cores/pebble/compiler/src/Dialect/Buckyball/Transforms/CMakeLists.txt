add_mlir_dialect_library(${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/smatmul/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/im2col/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/toint8/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/int2fp/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/lut/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/maxpool/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  ${BB_REPO_ROOT}/examples/balls/int8add/compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
