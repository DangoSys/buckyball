#ifndef MXQUANTBALL_OPS
#define MXQUANTBALL_OPS

include "Buckyball.td"

def MxquantOp : Buckyball_Op<"mxquant"> {
  let summary = "Finite FP32 to block32 MXFP8 quantization";
  let arguments = (ins I64:$inputBankId, I64:$outputBankId, I64:$iter);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` $iter "
                       "prop-dict attr-dict `:` type($inputBankId)";
}

def BankMxquantOp : Buckyball_Op<"bank_mxquant", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "FP32 to block32 MXFP8 quantization with virtual banks";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter prop-dict attr-dict `:` "
                       "type($inBank) type($outBank) type($iter)";
}

#endif
