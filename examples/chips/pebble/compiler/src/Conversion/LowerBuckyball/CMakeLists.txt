add_mlir_library(${BB_CHIP_TARGET_PREFIX}LowerBuckyballPass
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/BaseAssignPhysicalBankPatterns.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/PhysicalBankState.cpp
  ${BB_COMPILER_COMMON_DIR}/Conversion/LowerBuckyball/ReportBankUsagePass.cpp
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/matrix/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/im2col/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/fp2int/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  ${BB_REPO_ROOT}/examples/balls/int2fp/compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
  AssignPhysicalBanksPass.cpp
  LowerBuckyballPass.cpp
  LowerBuckyballToBankSSAPass.cpp
  MatrixMatmulToBankSSAPatterns.cpp
  Im2colMatmulToBankSSAPatterns.cpp
  MemTransposeToBankSSAPatterns.cpp

  LINK_LIBS PUBLIC
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyballTransforms
)

# CMake's archive rule appends members, so a chip switch can retain stale
# objects from the previously configured LowerBuckyballPass target.
add_custom_command(TARGET ${BB_CHIP_TARGET_PREFIX}LowerBuckyballPass PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E rm -f
    $<TARGET_FILE:${BB_CHIP_TARGET_PREFIX}LowerBuckyballPass>)
