//===-- MaxPoolBall.td - MaxPoolBall operations ------------*- tablegen -*-===//

#ifndef MAXPOOLBALL_OPS
#define MAXPOOLBALL_OPS

include "Buckyball.td"

def MaxPoolOp : Buckyball_Op<"maxpool"> {
  let summary = "Signed INT8 spatial max pooling";
  let description = [{
    Each bank row contains the 16 channels of one spatial position. The input
    and output tiles are square. inputBase selects the first input row.
    outputBase and outputStride place the result inside a larger resident
    feature-map tile. iter is the number of output spatial positions.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId, I64:$iter,
                       I64Attr:$inputSide, I64Attr:$outputSide,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding,
                       I64Attr:$inputBase, I64Attr:$outputBase,
                       I64Attr:$outputStride);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` $iter attr-dict `:` "
                       "type($inputBankId)";
}

def BankMaxPoolOp : Buckyball_Op<"bank_maxpool"> {
  let summary = "Signed INT8 spatial max pooling with virtual banks";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter,
                       I64Attr:$inputSide, I64Attr:$outputSide,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding,
                       I64Attr:$inputBase, I64Attr:$outputBase,
                       I64Attr:$outputStride);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter attr-dict `:` "
                       "type($inBank) type($outBank) type($iter)";
}

#endif // MAXPOOLBALL_OPS
