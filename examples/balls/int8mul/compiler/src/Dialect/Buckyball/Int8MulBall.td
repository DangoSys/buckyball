#ifndef INT8MULBALL_OPS
#define INT8MULBALL_OPS

include "Buckyball.td"

def Int8MulOp : Buckyball_Op<"int8mul"> {
  let summary = "Scaled INT8 per-channel broadcast multiply on physical banks";
  let arguments = (ins I64:$gateBankId, I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, F32:$ratio, I64:$gateRow);
  let assemblyFormat = "$gateBankId `,` $inputBankId `,` $outputBankId `,` "
                       "$iter `,` $ratio `,` $gateRow attr-dict `:` "
                       "type($gateBankId)";
}

def BankInt8MulOp : Buckyball_Op<"bank_int8mul"> {
  let summary = "Scaled INT8 per-channel broadcast multiply on virtual banks";
  let arguments = (ins I64:$gateBank, I64:$inputBank, I64:$outputBank,
                       I64:$iter, F32:$ratio, I64:$gateRow);
  let results = (outs I64:$outputBankOut);
  let assemblyFormat = "$gateBank $inputBank $outputBank $iter $ratio "
                       "$gateRow attr-dict `:` type($gateBank) "
                       "type($inputBank) type($outputBank) type($iter) "
                       "type($ratio) type($gateRow)";
}

#endif // INT8MULBALL_OPS
