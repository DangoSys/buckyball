//===-- GemminiBall.td - Gemmini Ball operations ------------------------===//

#ifndef GEMMINI_BALL_OPS
#define GEMMINI_BALL_OPS

include "Buckyball.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

def GemminiInstructionOp : Buckyball_Op<"gemmini"> {
  let summary = "Gemmini command encoded for the target ISA";
  let description = [{
    Emits one Gemmini command. mnemonic must name an ISA entry enabled by the
    selected target, and rs1/rs2 are the exact command operands defined by
    that entry. This covers GEMMINI_CONFIG, GEMMINI_PRELOAD,
    GEMMINI_COMPUTE_PRELOADED, GEMMINI_COMPUTE_ACCUMULATED, GEMMINI_LOOP_WS,
    and GEMMINI_LOOP_CONV_WS without hard-coding funct7 values in the IR.
  }];
  let arguments = (ins StrAttr:$mnemonic, I64:$rs1, I64:$rs2);
  let assemblyFormat = "$mnemonic $rs1 `,` $rs2 prop-dict attr-dict `:` type($rs1)";
}

def BankGemminiPreloadOp : Buckyball_Op<"bank_gemmini_preload"> {
  let summary = "BankSSA Gemmini preload";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$inBase,
                       I64:$outBase);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$inBank $outBank $iter $inBase $outBase prop-dict attr-dict `:` "
                       "type($inBank) type($outBank) type($iter) type($inBase) type($outBase)";
}

def BankGemminiComputePreloadedOp : Buckyball_Op<"bank_gemmini_compute_preloaded"> {
  let summary = "BankSSA Gemmini compute using preloaded state";
  let arguments = (ins I64:$aBank, I64:$bBank, I64:$outBank, I64:$iter,
                       I64:$aBase, I64:$bBase, I64:$outBase);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$aBank $bBank $outBank $iter $aBase $bBase $outBase prop-dict attr-dict `:` "
                       "type($aBank) type($bBank) type($outBank) type($iter) type($aBase) type($bBase) type($outBase)";
}

def BankGemminiComputeAccumulatedOp : Buckyball_Op<"bank_gemmini_compute_accumulated"> {
  let summary = "BankSSA Gemmini compute accumulating the output bank";
  let arguments = (ins I64:$aBank, I64:$bBank, I64:$outBank, I64:$iter,
                       I64:$aBase, I64:$bBase, I64:$outBase);
  let results = (outs I64:$outBankOut);
  let assemblyFormat = "$aBank $bBank $outBank $iter $aBase $bBase $outBase prop-dict attr-dict `:` "
                       "type($aBank) type($bBank) type($outBank) type($iter) type($aBase) type($bBase) type($outBase)";
}

def GemminiConvOp : Buckyball_Op<"gemmini_conv", [
  DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToAllocation",
    "bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues", "bufferize"]>]> {
  let summary = "NCHW convolution with INT8 HWIO matrix weights and scalar quantization scales";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
    AnyRankedTensor:$bias, I64Attr:$kh, I64Attr:$kw, I64Attr:$stride,
    I64Attr:$padding, F32Attr:$inputScale, F32Attr:$weightScale);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input `,` $weight `,` $bias prop-dict attr-dict `:` type($input) `,` type($weight) `,` type($bias) `->` type($output)";
}
def GemminiMemConvOp : Buckyball_Op<"gemmini_mem_conv"> {
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$weight, AnyMemRef:$bias,
    AnyMemRef:$output, I64Attr:$kh, I64Attr:$kw, I64Attr:$stride,
    I64Attr:$padding, F32Attr:$inputScale, F32Attr:$weightScale);
  let assemblyFormat = "$input `,` $weight `,` $bias `,` $output prop-dict attr-dict `:` type($input) `,` type($weight) `,` type($bias) `,` type($output)";
}

#endif // GEMMINI_BALL_OPS
