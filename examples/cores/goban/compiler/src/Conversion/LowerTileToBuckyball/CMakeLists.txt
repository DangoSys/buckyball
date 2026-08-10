add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
)
