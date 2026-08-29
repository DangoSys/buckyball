add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/AssignPhysicalBanksPass.cpp
  ${BB_assign_SOURCES}
  ${BB_REPO_ROOT}/examples/balls/smatmul/compiler/src/Conversion/LowerBuckyball/LowerBuckyballToBankSSAPatterns.cpp
  LowerBuckyballPass.cpp
  LowerBuckyballToBankSSAPass.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  MLIRLinalgDialect
)
