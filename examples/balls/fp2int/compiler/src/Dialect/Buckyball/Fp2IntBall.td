//===-- QuantBall.td - Fp2IntBall operations ----------------*- tablegen -*-===//
//
// Fp2IntBall operations for quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTBALL_OPS
#define QUANTBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// Fp2IntBall operations
//===----------------------------------------------------------------------===//

def Fp2IntOp : Buckyball_Op<"fp2int"> {
  let summary = "Quantize floating-point to fixed-point";
  let description = [{
    Quantizes FP32 words to packed INT8 words. The input and output bank
    group counts are independent: FP2INT streams input words in row/group
    order and packs each four words into one output word. The input stream
    must fill complete output rows.
    Corresponds to bb_fp2int instruction (funct7 from Core ballISA).
    Belongs to Fp2IntBall (ballName=Fp2IntBall; ballId is chip-config specific).
    ISA: bb_fp2int(bank_id, wr_bank_id, iter, da_addr)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$daAddr);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $daAddr attr-dict `:` "
                       "type($inputBankId)";
}

def BankFp2IntOp : Buckyball_Op<"bank_fp2int"> {
  let summary = "Quantization using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$daAddr);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $daAddr attr-dict `:` type($inBank) type($outBank) type($iter) type($daAddr)";
}

#endif // QUANTBALL_OPS
