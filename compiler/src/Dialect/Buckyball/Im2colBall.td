//===-- Im2colBall.td - Im2colBall operations --------------*- tablegen -*-===//
//
// Im2colBall operations for image-to-column transformation.
//
//===----------------------------------------------------------------------===//

#ifndef IM2COLBALL_OPS
#define IM2COLBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// Im2colBall operations
//===----------------------------------------------------------------------===//

def Im2colOp : Buckyball_Op<"im2col"> {
  let summary = "Image-to-column transformation for convolution";
  let description = [{
    Transforms image data into column format for efficient convolution.
    Corresponds to bb_im2col instruction (funct7=48).
    Belongs to Im2colBall (bid=3).
    ISA: bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, startcol)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$krow, I64:$kcol,
                       I64:$inrow, I64:$incol,
                       I64:$startrow, I64:$startcol);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$krow `,` $kcol `,` "
                       "$inrow `,` $incol `,` "
                       "$startrow `,` $startcol attr-dict `:` "
                       "type($inputBankId)";
}

#endif // IM2COLBALL_OPS
