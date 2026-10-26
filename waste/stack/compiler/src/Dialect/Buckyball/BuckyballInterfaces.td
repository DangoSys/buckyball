//===-- BuckyballInterfaces.td - Bank lowering interfaces --*- tablegen -*-===//
//
// The framework owns bank lowering. A Ball takes part by implementing these
// interfaces on its operations.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_INTERFACES
#define BUCKYBALL_INTERFACES

include "mlir/IR/OpBase.td"

def BankSSALoweringInterface : OpInterface<"BankSSALowering"> {
  let cppNamespace = "::buddy::buckyball";
  let description = [{
    A memref-level Ball operation that rewrites itself into explicit bank-SSA
    operations: bank allocation, mvin, the bank-level Ball operation and mvout.
    Lowered by `lower-buckyball-to-bank-ssa`.
  }];
  let methods = [
    InterfaceMethod<"Replaces this operation with bank-SSA operations.",
      "::mlir::LogicalResult", "lowerToBankSSA",
      (ins "::mlir::PatternRewriter &":$rewriter)>
  ];
}

def PhysicalBankAssignmentInterface : OpInterface<"PhysicalBankAssignment"> {
  let cppNamespace = "::buddy::buckyball";
  let description = [{
    A bank-SSA Ball operation that rewrites itself into its physical-bank
    operation once `assign-physical-banks` has mapped its bank handles.
  }];
  let methods = [
    InterfaceMethod<"Replaces this operation with its physical operation.",
      "::mlir::LogicalResult", "assignPhysicalBanks",
      (ins "::mlir::PatternRewriter &":$rewriter,
           "::mlir::buddy::PhysicalBankState &":$state)>
  ];
}

#endif // BUCKYBALL_INTERFACES
