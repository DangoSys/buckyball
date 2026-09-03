#ifndef INT8ADDBALL_OPS
#define INT8ADDBALL_OPS

include "Buckyball.td"

def Int8AddOp : Buckyball_Op<"int8add"> {
  let summary = "Scaled INT8 addition on physical banks";
  let arguments = (ins I64:$lhsBankId, I64:$rhsBankId, I64:$outputBankId,
                       I64:$iter, F32:$lhsRatio, F32:$rhsRatio,
                       BoolAttr:$relu);
  let assemblyFormat = "$lhsBankId `,` $rhsBankId `,` $outputBankId `,` $iter `,` "
                       "$lhsRatio `,` $rhsRatio attr-dict `:` type($lhsBankId)";
}

def BankInt8AddOp : Buckyball_Op<"bank_int8add"> {
  let summary = "Scaled INT8 addition on virtual banks";
  let arguments = (ins I64:$lhsBank, I64:$rhsBank, I64:$outputBank,
                       I64:$iter, F32:$lhsRatio, F32:$rhsRatio,
                       BoolAttr:$relu);
  let results = (outs I64:$outputBankOut);
  let assemblyFormat = "$lhsBank $rhsBank $outputBank $iter $lhsRatio $rhsRatio "
                       "attr-dict `:` type($lhsBank) type($rhsBank) "
                       "type($outputBank) type($iter) type($lhsRatio)";
}

#endif // INT8ADDBALL_OPS
