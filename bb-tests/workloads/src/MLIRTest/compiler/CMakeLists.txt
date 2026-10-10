set(BUDDY_MLIR_DIR ${BUCKYBALL_REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/${BUCKYBALL_WORKLOAD_CHIP}/bin)
set(BUDDY_OPT ${BUDDY_BINARY_DIR}/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_BINARY_DIR}/buddy-translate)
set(BUDDY_LLC ${BUDDY_BINARY_DIR}/buddy-llc)
set(LLVM_BUILD_DIR ${BUDDY_MLIR_DIR}/llvm/build)

option(BUCKYBALL_STABLE "Use stable LLVM Buckyball extensions" OFF)
if(BUCKYBALL_STABLE)
  set(BUCKYBALL_RISCV_MATTR "+xbbpebble,+xbuckyball,+m,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL "--lower-buckyball=stable=true")

  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      "--lower-bank-ssa-to-intrinsics=stable=true")

else()
  set(BUCKYBALL_RISCV_MATTR "+xbuckyball,+m,+D")
  set(BUCKYBALL_LOWER_BUCKYBALL -lower-buckyball)

  set(BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS
      -lower-bank-ssa-to-intrinsics)

endif()
message(STATUS "Buckyball RISC-V mattr: ${BUCKYBALL_RISCV_MATTR}")
