#ifndef MATADDBALL_OPS
#define MATADDBALL_OPS

include "Buckyball.td"

def MatAddOp : Buckyball_Op<"matadd"> {
  let summary = "Add matching INT32 physical-bank lines";
  let arguments = (ins I64:$aBankId, I64:$bBankId, I64:$cBankId, I64:$iter);
  let assemblyFormat = "$aBankId `,` $bBankId `,` $cBankId `,` $iter attr-dict `:` "
                       "type($aBankId)";
}

def BankMatAddOp : Buckyball_Op<"bank_matadd"> {
  let summary = "Add matching INT32 virtual-bank lines";
  let arguments = (ins I64:$aBank, I64:$bBank, I64:$cBank, I64:$iter);
  let results = (outs I64:$cBankOut);
  let assemblyFormat = "$aBank $bBank $cBank $iter attr-dict `:` "
                       "type($aBank) type($bBank) type($cBank) type($iter)";
}

#endif // MATADDBALL_OPS
