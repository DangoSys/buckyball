add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp
  AssignPhysicalBanksPass.cpp
  LowerBuckyballPass.cpp
  LowerBuckyballToBankSSAPass.cpp
  Patterns/ToyAssignPhysicalBankPatterns.cpp
  Patterns/ToyLowerBuckyballToBankSSAPatterns.cpp
  Patterns/ToyLowerBuckyballToLLVMPatterns.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  )

add_custom_command(TARGET ${BB_CORE_TARGET_PREFIX}LowerBuckyballPass PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E rm -f
    $<TARGET_FILE:${BB_CORE_TARGET_PREFIX}LowerBuckyballPass>)
