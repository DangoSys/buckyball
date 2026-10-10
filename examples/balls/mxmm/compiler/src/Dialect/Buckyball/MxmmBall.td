#ifndef MATMULBALL_OPS
#define MATMULBALL_OPS

include "Buckyball.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

def MXFP8QuantOp : Buckyball_Op<"mxfp8_quant", [NoMemoryEffect,
  DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToAllocation",
    "bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues", "bufferize"]>]> {
  let summary = "Integer RNE activation quantization into aligned MXFP8 panels";
  let description = [{
    The input is an immutable finite FP32 matrix. Each panel contains all
    row-major E4M3 codes followed by row-major block32 E8M0 scales. The final
    K panel uses its actual remaining width. Unused bank-window padding is
    undefined and must not be consumed as codes or scales. Quantization does
    not read frm or modify fflags. This operation is not speculatable.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$tile_rows,
    I64Attr:$tile_k, I64Attr:$bank_bytes);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat = "$input prop-dict attr-dict `:` type($input) `->` type($output)";
}

def MXFP8MatmulOp : Buckyball_Op<"mxfp8_matmul", [
  DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToAllocation",
    "bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues", "bufferize"]>]> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight, I64Attr:$reduction_k, I64Attr:$tile_k, I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$bank_bytes, I64Attr:$activation_tile_k);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat = "$input `,` $weight prop-dict attr-dict `:` type($input) `,` type($weight) `->` type($output)";
}
def MXFP8MemMatmulOp : Buckyball_Op<"mxfp8_mem_matmul"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$input,
    Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$weight,
    Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$output, I64Attr:$reduction_k, I64Attr:$tile_k, I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$bank_bytes, I64Attr:$activation_tile_k);
  let assemblyFormat = "$input `,` $weight `,` $output prop-dict attr-dict `:` type($input) `,` type($weight) `,` type($output)";
}
def MXFP8SharedPanelOp : Buckyball_Op<"mxfp8_shared_panel", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Compute one original MXFP8 output panel into a live shared bank";
  let arguments = (ins MemRefRankOf<[I8], [1]>:$input,
    MemRefRankOf<[I8], [1]>:$weight, I64:$bank,
    I64Attr:$rows, I64Attr:$columns, I64Attr:$reduction_k,
    I64Attr:$tile_k, I64Attr:$bank_bytes, I64Attr:$shared_bytes);
  let results = (outs I64:$bank_out);
  let assemblyFormat = "$input `,` $weight `,` $bank prop-dict attr-dict `:` "
    "type($input) `,` type($weight) `,` type($bank)";
}
def MXFP8SharedWindowOp : Buckyball_Op<"mxfp8_shared_window", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Compute a shared output panel from one complete MXFP8 activation bank";
  let arguments = (ins MemRefRankOf<[I8], [1]>:$input,
    MemRefRankOf<[I8], [1]>:$weight, I64:$bank,
    I64Attr:$rows, I64Attr:$columns, I64Attr:$reduction_k,
    I64Attr:$tile_k, I64Attr:$bank_bytes, I64Attr:$shared_bytes);
  let results = (outs I64:$bank_out);
  let assemblyFormat = "$input `,` $weight `,` $bank prop-dict attr-dict `:` "
    "type($input) `,` type($weight) `,` type($bank)";
}
def MXFP8Op : Buckyball_Op<"mxfp8"> {
  let summary = "MxmmBall compute";
  let description = [{
    ISA: MXMM_MXFP8. `config` contains rows, cols, and k.
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config `,` $first `,` $last `,` $outputBase "
                       "prop-dict attr-dict `:` "
                       "type($op1BankId)";
}

def BankMXFP8Op : Buckyball_Op<"bank_mxfp8"> {
  let summary = "MxmmBall compute with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $config $first $last $outputBase "
      "prop-dict attr-dict `:` "
      "type($op1Bank) type($op2Bank) type($wrBank) type($config)";
}

def MXFP8WindowOp : Buckyball_Op<"mxfp8_window"> {
  let summary = "MXFP8 multiplication from a complete activation row window";
  let arguments = (ins I64:$op1BankId, I64:$op2BankId, I64:$resultBankId,
    I64:$rows, I64:$cols, I64:$count, I64:$fullK, I64:$startK,
    I1:$first, I1:$last, I64:$outputBase);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
    "$rows `,` $cols `,` $count `,` $fullK `,` $startK `,` "
    "$first `,` $last `,` $outputBase prop-dict attr-dict `:` type($op1BankId)";
}

def BankMXFP8WindowOp : Buckyball_Op<"bank_mxfp8_window", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "MXFP8 activation row window with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank,
    I64:$rows, I64:$cols, I64:$count, I64:$fullK, I64:$startK,
    I1:$first, I1:$last, I64:$outputBase);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat = "$op1Bank $op2Bank $wrBank $rows $cols $count "
    "$fullK $startK $first $last $outputBase prop-dict attr-dict `:` type($op1Bank)";
}

def FP32MatmulOp : Buckyball_Op<"fp32_matmul", [
  DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferizesToAllocation",
    "bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues", "bufferize"]>]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
    OptionalAttr<UnitAttr>:$rhs_transposed, BoolAttr:$fused);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs prop-dict attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}
def FP32MemMatmulOp : Buckyball_Op<"fp32_mem_matmul"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$lhs,
    Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$rhs,
    Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$output, BoolAttr:$fused);
  let assemblyFormat = "$lhs `,` $rhs `,` $output prop-dict attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
def FP32Op : Buckyball_Op<"matmul_f32"> {
  let summary = "MxmmBall compute";
  let description = [{
    ISA: MXMM_FMA32 / MXMM_F32. `config` contains rows, cols, and k.
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase, BoolAttr:$fused);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config `,` $first `,` $last `,` $outputBase "
                       "prop-dict attr-dict `:` "
                       "type($op1BankId)";
}

def BankFP32Op : Buckyball_Op<"bank_matmul_f32"> {
  let summary = "MxmmBall compute with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase, BoolAttr:$fused);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $config $first $last $outputBase "
      "prop-dict attr-dict `:` "
      "type($op1Bank) type($op2Bank) type($wrBank) type($config)";
}



#endif
