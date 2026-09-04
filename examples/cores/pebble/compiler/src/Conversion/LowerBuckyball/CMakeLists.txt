add_mlir_library(${BB_CORE_TARGET_PREFIX}LowerBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/AssignPhysicalBanksPass.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportCycleEstimatePass.cpp
  ${BB_assign_SOURCES}
  ${BB_REPO_ROOT}/examples/balls/smatmul/compiler/src/Conversion/LowerBuckyball/LowerBuckyballToBankSSAPatterns.cpp
  CoreBankSSALowering.cpp
  LowerBuckyballPass.cpp
  LowerBuckyballToBankSSAPass.cpp
  MegaKernelToBankSSAPatterns.cpp
  ResidentConvRegionToBankSSAPatterns.cpp
  PointConvRegionToBankSSAPatterns.cpp
  MegaConv2dToBankSSAPatterns.cpp
  MemTransposeToBankSSAPatterns.cpp
  QuantizeTensorToBankSSAPatterns.cpp

  LINK_LIBS PUBLIC
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
)

# CMake's archive rule appends members, so a chip switch can retain stale
# objects from the previously configured LowerBuckyballPass target.
add_custom_command(TARGET ${BB_CORE_TARGET_PREFIX}LowerBuckyballPass PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E rm -f
    $<TARGET_FILE:${BB_CORE_TARGET_PREFIX}LowerBuckyballPass>)
