//===-- Mxfp2IntBall.td - MXFP to INT8 dequantization operations --------===//

#ifndef MXFP2INT_BALL_OPS
#define MXFP2INT_BALL_OPS

include "Buckyball.td"

def Mxfp2IntOp : Buckyball_Op<"mxfp2int"> {
  let summary = "Dequantize MXFP blocks to INT8";
  let description = [{
    Reads MXFP blocks from inputBankId, writes two INT8 rows per block to
    outputBankId, and uses special as rs2 for format and rounding controls.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId, I64:$iter,
                       I64:$special);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` $iter `,` $special attr-dict `:` type($inputBankId)";
}

def BankMxfp2IntOp : Buckyball_Op<"bank_mxfp2int"> {
  let summary = "MXFP to INT8 dequantization using bank-SSA handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$special);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter $special attr-dict `:` type($inBank) type($outBank) type($iter) type($special)";
}

#endif // MXFP2INT_BALL_OPS
