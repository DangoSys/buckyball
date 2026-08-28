add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
  CoreTileLowering.cpp
  ${BB_REPO_ROOT}/examples/balls/smatmul/compiler/src/Conversion/LowerTileToBuckyball/TileMatMulPatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Conversion/LowerTileToBuckyball/TileTransposePatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/relu/compiler/src/Conversion/LowerTileToBuckyball/TileReluPatterns.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
)
