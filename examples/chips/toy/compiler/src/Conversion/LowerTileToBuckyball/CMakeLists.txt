add_mlir_library(${BB_CHIP_TARGET_PREFIX}LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyball
  MLIRLinalgDialect
  )
