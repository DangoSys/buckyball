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
    Belongs to Im2colBall (bid=3).
    ISA: bb_im2col_ex(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, startcol, col_step)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$krow, I64:$kcol,
                       I64:$inrow, I64:$incol,
                       I64:$startrow, I64:$startcol,
                       I64:$colStep);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$krow `,` $kcol `,` "
                       "$inrow `,` $incol `,` "
                       "$startrow `,` $startcol `,` $colStep attr-dict `:` "
                       "type($inputBankId)";
}

def BankIm2colOp : Buckyball_Op<"bank_im2col"> {
  let summary = "Im2col using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$krow, I64:$kcol,
                       I64:$inrow, I64:$incol, I64:$startrow, I64:$startcol,
                       I64:$colStep);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $krow $kcol $inrow $incol $startrow $startcol $colStep attr-dict `:` type($inBank) type($outBank) type($krow) type($kcol) type($inrow) type($incol) type($startrow) type($startcol) type($colStep)";
}

def Im2colIntrOp : Buckyball_IntrOpBase<"im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // IM2COLBALL_OPS
