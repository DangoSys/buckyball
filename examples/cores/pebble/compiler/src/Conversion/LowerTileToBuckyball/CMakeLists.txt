add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerTileToBuckyball/QuantizedKernelPatterns.cpp
  CoreTileLowering.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Conversion/LowerTileToBuckyball/TileTransposePatterns.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
)
