//===-- ToInt8Ball.td - ToInt8Ball operations ----------------*- tablegen -*-===//
//
// ToInt8Ball operations for quantization.
//
//===----------------------------------------------------------------------===//

#ifndef TOINT8BALL_OPS
#define TOINT8BALL_OPS

include "Buckyball.td"

def QuantF32ToI8Op : Buckyball_Op<"quant_f32_to_i8"> {
  let summary = "Per-tensor FP32 to INT8 quantization";
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, F32:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` $iter `,` $scale "
                       "attr-dict `:` type($inputBankId)";
}

def BankQuantF32ToI8Op : Buckyball_Op<"bank_quant_f32_to_i8"> {
  let summary = "Per-tensor FP32 to INT8 quantization with virtual banks";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, F32:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter $scale attr-dict `:` "
                       "type($inBank) type($outBank) type($iter) type($scale)";
}

def QuantI32ToI8Op : Buckyball_Op<"quant_i32_to_i8"> {
  let summary = "Per-channel INT32 to INT8 requantization";
  let description = [{
    The output is a dense input stream placed in a strided 2-D bank tile.
    iter/4 must equal outputWidth*outputHeight.
  }];
  let arguments = (ins I64:$inputBankId, I64:$scaleBankId,
                       I64:$outputBankId, I64:$iter, I64Attr:$outputBase,
                       I64Attr:$outputWidth, I64Attr:$outputHeight,
                       I64Attr:$outputStride,
                       BoolAttr:$relu);
  let assemblyFormat = "$inputBankId `,` $scaleBankId `,` $outputBankId `,` "
                       "$iter attr-dict `:` type($inputBankId)";
}

def BankQuantI32ToI8Op : Buckyball_Op<"bank_quant_i32_to_i8"> {
  let summary = "Per-channel INT32 to INT8 requantization with virtual banks";
  let arguments = (ins I64:$inBank, I64:$scaleBank, I64:$outBank,
                       I64:$iter, I64Attr:$outputBase,
                       I64Attr:$outputWidth, I64Attr:$outputHeight,
                       I64Attr:$outputStride, BoolAttr:$relu);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $scaleBank $outBank $iter attr-dict `:` "
                       "type($inBank) type($scaleBank) type($outBank) type($iter)";
}

#endif // TOINT8BALL_OPS
