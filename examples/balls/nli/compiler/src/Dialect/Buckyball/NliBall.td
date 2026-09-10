#ifndef NLIBALL_OPS
#define NLIBALL_OPS

include "Buckyball.td"

def NliOp : Buckyball_Op<"nli"> {
  let summary = "Apply a non-uniform linear interpolation (NLI) segment table";
  let arguments = (ins I64:$inputBankId, I64:$tableBankId,
                       I64:$outputBankId, I64:$iter);
  let assemblyFormat = "$inputBankId `,` $tableBankId `,` $outputBankId `,` "
                       "$iter attr-dict `:` type($inputBankId)";
}

def BankNliOp : Buckyball_Op<"bank_nli"> {
  let summary = "Apply an NLI segment table with virtual banks";
  let arguments = (ins I64:$inBank, I64:$tableBank, I64:$outBank, I64:$iter);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $tableBank $outBank $iter attr-dict `:` "
                       "type($inBank) type($tableBank) type($outBank) type($iter)";
}

#endif // NLIBALL_OPS
