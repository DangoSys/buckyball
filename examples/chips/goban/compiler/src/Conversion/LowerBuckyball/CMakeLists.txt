add_mlir_library(${BB_CHIP_TARGET_PREFIX}LowerBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  AssignPhysicalBanksPass.cpp
  LowerBuckyballPass.cpp
  LowerBuckyballToBankSSAPass.cpp

  LINK_LIBS PUBLIC
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyballTransforms
)
