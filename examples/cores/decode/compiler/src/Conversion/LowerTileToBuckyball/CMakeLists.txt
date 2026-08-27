add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
)
