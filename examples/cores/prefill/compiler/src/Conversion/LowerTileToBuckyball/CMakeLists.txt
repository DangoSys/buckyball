add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp
  ${BB_REPO_ROOT}/examples/balls/smatmul/compiler/src/Conversion/LowerTileToBuckyball/TileMatMulPatterns.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
)
