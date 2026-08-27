//===-- ReluBall.td - ReluBall operations ------------------*- tablegen -*-===//

#ifndef RELUBALL_OPS
#define RELUBALL_OPS

include "Buckyball.td"

def ReluMatrixOp : Buckyball_Op<"relu_matrix"> {
  let summary = "Logical INT32 ReLU matrix tile";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def ReluOp : Buckyball_Op<"relu"> {
  let summary = "In-place INT32 ReLU on one physical bank";
  let arguments = (ins I64:$bankId, I64:$group, I64:$iter, I64:$stride);
  let assemblyFormat = "$bankId `,` $group `,` $iter `,` $stride attr-dict `:` "
                       "type($bankId)";
}

def BankReluOp : Buckyball_Op<"bank_relu"> {
  let summary = "In-place INT32 ReLU with a virtual bank handle";
  let arguments = (ins I64:$bank, I64:$group, I64:$iter, I64:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$bank $group $iter $stride attr-dict `:` "
                       "type($bank) type($group) type($iter) type($stride)";
}

#endif // RELUBALL_OPS
