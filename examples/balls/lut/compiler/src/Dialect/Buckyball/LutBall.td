#ifndef LUTBALL_OPS
#define LUTBALL_OPS

include "Buckyball.td"

def LutOp : Buckyball_Op<"lut"> {
  let summary = "Apply a 256-entry INT8 lookup table";
  let arguments = (ins I64:$inputBankId, I64:$lutBankId,
                       I64:$outputBankId, I64:$iter);
  let assemblyFormat = "$inputBankId `,` $lutBankId `,` $outputBankId `,` "
                       "$iter attr-dict `:` type($inputBankId)";
}

def BankLutOp : Buckyball_Op<"bank_lut"> {
  let summary = "Apply a 256-entry INT8 lookup table with virtual banks";
  let arguments = (ins I64:$inBank, I64:$lutBank, I64:$outBank, I64:$iter);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $lutBank $outBank $iter attr-dict `:` "
                       "type($inBank) type($lutBank) type($outBank) type($iter)";
}

#endif // LUTBALL_OPS
