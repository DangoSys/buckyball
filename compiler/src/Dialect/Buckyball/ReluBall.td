//===-- ReluBall.td - ReluBall operations ------------------*- tablegen -*-===//
//
// ReluBall operations for ReLU activation function.
//
//===----------------------------------------------------------------------===//

#ifndef RELUBALL_OPS
#define RELUBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// ReluBall operations
//===----------------------------------------------------------------------===//

def ReluOp : Buckyball_Op<"relu"> {
  let summary = "ReLU activation function";
  let description = [{
    Applies ReLU (Rectified Linear Unit) activation: max(0, x).
    Corresponds to bb_relu instruction (funct7=50).
    Belongs to ReluBall (bid=1).
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$depth, I64:$stride);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$depth `,` $stride attr-dict `:` "
                       "type($inputBankId)";
}

#endif // RELUBALL_OPS
