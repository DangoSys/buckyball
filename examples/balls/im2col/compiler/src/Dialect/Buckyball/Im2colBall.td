//===-- Im2colBall.td - Im2colBall operations --------------*- tablegen -*-===//
//
// Im2colBall operations for image-to-column transformation.
//
//===----------------------------------------------------------------------===//

#ifndef IM2COLBALL_OPS
#define IM2COLBALL_OPS

include "Buckyball.td"

//===----------------------------------------------------------------------===//
// Im2colBall operations
//===----------------------------------------------------------------------===//

// Memref-level im2col+gemm (like smatmul_matmul). Chip bank-ssa decides
// f32 quant / Bank* composition; this op carries no quant policy.
def Im2colMatmulOp : Buckyball_Op<"im2col_matmul"> {
  let summary = "Packed-plane im2col then matrix gemm";
  let description = [{
    One cin-plane im2col, then gemm over all N columns of `filter`.
    `n` is OC (padded to a multiple of 16, 16..4096). `input` is one packed
    HxW plane (bank rows x 16). `filter`/`output` are KxN and MxN.
    Asym pad co-design: padding=max(lo,hi), startRow/Col=padding-lo.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$filter,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64Attr:$inSize,
                       I64Attr:$ksize,
                       I64Attr:$n,
                       DefaultValuedAttr<I64Attr, "1">:$stride,
                       DefaultValuedAttr<I64Attr, "0">:$padding,
                       DefaultValuedAttr<I64Attr, "0">:$startRow,
                       DefaultValuedAttr<I64Attr, "0">:$startCol,
                       DefaultValuedAttr<I64Attr, "-1">:$dwAddr,
                       DefaultValuedAttr<I64Attr, "0">:$dwBytes,
                       DefaultValuedAttr<BoolAttr, "false">:$perChannel);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

// Standard conv: nCin input planes -> one im2col+gemm with K = nCin*K^2.
def Im2colFatMatmulOp : Buckyball_Op<"im2col_fat_matmul"> {
  let summary = "Multi-plane im2col then fat gemm over nCin input channels";
  let description = [{
    Stack $nCin input planes as [nCin*inRows, 16] and filters as
    [nCin*bRows, n]. Im2col each plane, concatenate on K, then one
    (wins x nCin*K^2) @ (nCin*K^2 x n) gemm. $n is OC padded to 16.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$filter,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64Attr:$inSize,
                       I64Attr:$ksize,
                       I64Attr:$n,
                       I64Attr:$nCin,
                       DefaultValuedAttr<I64Attr, "1">:$stride,
                       DefaultValuedAttr<I64Attr, "0">:$padding,
                       DefaultValuedAttr<I64Attr, "0">:$startRow,
                       DefaultValuedAttr<I64Attr, "0">:$startCol,
                       DefaultValuedAttr<I64Attr, "-1">:$dwAddr,
                       DefaultValuedAttr<I64Attr, "0">:$dwBytes,
                       DefaultValuedAttr<BoolAttr, "false">:$perChannel);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

// Depthwise: n channels (1..16) share one fat GEMM, not per-channel GEMV.
// Input planes stacked as [n*inRows, 16]; filter is block-diag [n*bRows, 16].
def Im2colDepthwiseMatmulOp : Buckyball_Op<"im2col_depthwise_matmul"> {
  let summary = "Multi-plane im2col then fat depthwise gemm (N=lane)";
  let description = [{
    For channel tile size $n (1..16):
      im2col each plane, concatenate on K, then (wins x n*K^2) @ (n*K^2 x n)
    with block-diagonal weights. Lane-full GEMM; not GEMV.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$filter,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64Attr:$inSize,
                       I64Attr:$ksize,
                       I64Attr:$n,
                       DefaultValuedAttr<I64Attr, "1">:$stride,
                       DefaultValuedAttr<I64Attr, "0">:$padding,
                       DefaultValuedAttr<I64Attr, "0">:$startRow,
                       DefaultValuedAttr<I64Attr, "0">:$startCol,
                       DefaultValuedAttr<I64Attr, "-1">:$dwAddr,
                       DefaultValuedAttr<I64Attr, "0">:$dwBytes,
                       DefaultValuedAttr<BoolAttr, "false">:$perChannel);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

def Im2colOp : Buckyball_Op<"im2col"> {
  let summary = "Image-to-column transformation for convolution";
  let description = [{
    Transforms image data into column format for efficient convolution.
    Each input bank row is one spatial position with 16 INT8 channels.
    inputBase selects the first spatial row and lane selects one channel.
    windowStart/windowCount select a contiguous output-window range and rebase
    it to row zero in the output bank.
    Legal range is defined in examples/balls/im2col/configs/default.toml.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$ksize, I64:$stride, I64:$padding,
                       I64:$inputBase, I64:$lane,
                       I64Attr:$startRow, I64Attr:$startCol,
                       I64Attr:$windowStart, I64Attr:$windowCount);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $ksize `,` $stride `,` $padding `,` "
                       "$inputBase `,` $lane attr-dict `:` "
                       "type($inputBankId)";
}

def BankIm2colOp : Buckyball_Op<"bank_im2col"> {
  let summary = "Im2col using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$ksize,
                       I64:$stride, I64:$padding,
                       I64:$inputBase, I64:$lane,
                       I64Attr:$startRow, I64Attr:$startCol,
                       I64Attr:$windowStart, I64Attr:$windowCount);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $ksize $stride $padding $inputBase $lane "
      "attr-dict `:` type($inBank) type($outBank) type($iter) type($ksize) "
      "type($stride) type($padding) type($inputBase) type($lane)";
}

def Im2colIntrOp : Buckyball_IntrOpBase<"im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // IM2COLBALL_OPS
