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

// Memref-level im2col+gemm (like matrix_matmul). Chip bank-ssa decides
// f32 quant / Bank* composition; this op carries no quant policy.
def Im2colMatmulOp : Buckyball_Op<"im2col_matmul"> {
  let summary = "Packed-plane im2col then matrix gemm";
  let description = [{
    Computes one cin-plane gemm tile:
      output[M,16] from im2col(input) x filter, with matrix N = $n (1..16).
    `input` is a packed HxW plane (bank rows x 16), `filter` is Kx16 with
    K = ksize*ksize (row-padded to a multiple of 16). Stride/padding are
    im2col ISA fields; pebble currently only emits stride=1 padding=0.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$filter,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64Attr:$inSize,
                       I64Attr:$ksize,
                       I64Attr:$n,
                       DefaultValuedAttr<I64Attr, "1">:$stride,
                       DefaultValuedAttr<I64Attr, "0">:$padding);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

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
