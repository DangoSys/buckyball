//===-- Im2colBall.td - Im2colBall operations --------------*- tablegen -*-===//
//
// Im2colBall operations for image-to-column transformation.
//
//===----------------------------------------------------------------------===//

#ifndef IM2COLBALL_OPS
#define IM2COLBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// Im2colBall operations
//===----------------------------------------------------------------------===//

def Im2colOp : Buckyball_Op<"im2col"> {
  let summary = "Image-to-column transformation for convolution";
  let description = [{
    Transforms image data into column format for efficient convolution.
    Corresponds to bb_im2col instruction (funct7=48).
    Ball id (BID) is chip-specific via balldomain TOML; do not hardcode.
    ISA: bb_im2col(op1_bank_id, wr_bank_id, iter, ksize, stride, padding)
    Legal range is defined in examples/balls/im2col/configs/default.toml
    (maxIter / maxKSize / maxPadding); out of range is illegal fail-hard.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$ksize, I64:$stride, I64:$padding);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $ksize `,` $stride `,` $padding attr-dict `:` "
                       "type($inputBankId)";
}

def BankIm2colOp : Buckyball_Op<"bank_im2col"> {
  let summary = "Im2col using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$ksize,
                       I64:$stride, I64:$padding);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $ksize $stride $padding attr-dict `:` type($inBank) type($outBank) type($iter) type($ksize) type($stride) type($padding)";
}

def Im2colIntrOp : Buckyball_IntrOpBase<"im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // IM2COLBALL_OPS
