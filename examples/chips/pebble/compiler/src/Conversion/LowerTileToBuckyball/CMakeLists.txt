add_mlir_library(${BB_CHIP_TARGET_PREFIX}LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp
  ${BB_REPO_ROOT}/examples/balls/matrix/compiler/src/Conversion/LowerTileToBuckyball/TileMatMulPatterns.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
)
