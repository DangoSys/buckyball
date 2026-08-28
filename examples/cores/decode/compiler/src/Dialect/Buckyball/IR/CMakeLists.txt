add_mlir_dialect_library(${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/IR/BuckyballDialect.cpp
  ${BB_TARGET_REGISTRY_SOURCE}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)

target_include_directories(${BB_CORE_TARGET_PREFIX}BuddyBuckyball PUBLIC
  ${BB_COMPILER_COMMON_DIR}
  ${BB_COMPILER_COMMON_DIR}/Dialect
  ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/../../..
)
