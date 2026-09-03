//===-- Int2FpBall.td - Int2FpBall operations --------------*- tablegen -*-===//
//
// Int2FpBall operations for Mega Kernel output conversion.
//
//===----------------------------------------------------------------------===//

#ifndef INT2FPBALL_OPS
#define INT2FPBALL_OPS

include "Buckyball.td"

//===----------------------------------------------------------------------===//
// Int2FpBall operations
//===----------------------------------------------------------------------===//

def Int32ToFp32Op : Buckyball_Op<"int32_to_fp32"> {
  let summary = "Convert INT32 accumulators to FP32";
  let description = [{
    Converts four INT32 values per input row to FP32 and multiplies them by
    the matching values from a 16-channel FP32 scale bank. Scale rows repeat
    every four accumulator rows.
  }];
  let arguments = (ins I64:$inputBankId, I64:$scaleBankId,
                       I64:$outputBankId, I64:$iter, BoolAttr:$relu);
  let assemblyFormat = "$inputBankId `,` $scaleBankId `,` $outputBankId `,` "
                       "$iter attr-dict `:` "
                       "type($inputBankId)";
}

def BankInt32ToFp32Op : Buckyball_Op<"bank_int32_to_fp32"> {
  let summary = "INT32 to FP32 conversion with virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$scaleBank, I64:$outBank,
                       I64:$iter, BoolAttr:$relu);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $scaleBank $outBank $iter attr-dict `:` "
      "type($inBank) type($scaleBank) type($outBank) type($iter)";
}

#endif // INT2FPBALL_OPS
