//===-- Int2FpBall.td - Int2FpBall operations --------------*- tablegen -*-===//
//
// Int2FpBall operations for dequantization / requantization.
//
//===----------------------------------------------------------------------===//

#ifndef DEQUANTBALL_OPS
#define DEQUANTBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// Int2FpBall operations
//===----------------------------------------------------------------------===//

class Int2FpBaseOp<string mnemonic, string summaryText> : Buckyball_Op<mnemonic> {
  let summary = summaryText;
  let description = [{
    Dequantizes INT32 accumulators to FP32 using Da and Dw scales in MMIO.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$daAddr, I64:$dwAddr);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $daAddr `,` $dwAddr attr-dict `:` "
                       "type($inputBankId)";
}

def Int2FpTensorOp : Int2FpBaseOp<"int2fp_tensor",
    "Per-tensor INT32 accumulator dequantization">;

def Int2FpChannelOp : Int2FpBaseOp<"int2fp_channel",
    "Per-output-channel INT32 accumulator dequantization">;

class BankInt2FpBaseOp<string mnemonic, string summaryText> : Buckyball_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$daAddr,
                       I64:$dwAddr);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $daAddr $dwAddr attr-dict `:` type($inBank) type($outBank) type($iter) type($daAddr) type($dwAddr)";
}

def BankInt2FpTensorOp : BankInt2FpBaseOp<"bank_int2fp_tensor",
    "Per-tensor dequantization using virtual bank handles">;

def BankInt2FpChannelOp : BankInt2FpBaseOp<"bank_int2fp_channel",
    "Per-output-channel dequantization using virtual bank handles">;

#endif // DEQUANTBALL_OPS
