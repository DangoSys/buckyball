//===-- Buckyball.td - Buckyball dialect operations -----*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_OPS
#define BUCKYBALL_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Dialect and common operations
//===----------------------------------------------------------------------===//

def Buckyball_Dialect : Dialect {
  let name = "buckyball";
  let cppNamespace = "::buddy::buckyball";
  let summary = "Dialect for RISC-V Buckyball NPU extension";
  let description = [{
    Buckyball is an NPU Design Framework with composable Ball units.
    Each Ball is a specialized compute unit that can be independently
    developed and combined with others.
  }];
}

class Buckyball_Op<string mnemonic, list<Trait> traits = []> :
  Op<Buckyball_Dialect, mnemonic, traits> {}

def MvinOp : Buckyball_Op<"mvin"> {
  let summary = "Load data from memory to scratchpad bank";
  let description = [{
    Loads data from DRAM to scratchpad bank.
    Corresponds to bb_mvin instruction (funct7=33).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       I64:$addr, I64:$depth, I64:$stride);
  let assemblyFormat = "$input $addr $depth $stride attr-dict `:` "
                       "type($input) type($addr) type($depth) type($stride)";
}

def MvoutOp : Buckyball_Op<"mvout"> {
  let summary = "Store data from scratchpad bank to memory";
  let description = [{
    Stores data from scratchpad bank to DRAM.
    Corresponds to bb_mvout instruction (funct7=16).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output,
                       I64:$addr, I64:$depth, I64:$stride);
  let assemblyFormat = "$output $addr $depth $stride attr-dict `:` "
                       "type($output) type($addr) type($depth) type($stride)";
}

def MsetOp : Buckyball_Op<"mset"> {
  let summary = "Allocate or release scratchpad bank memory";
  let description = [{
    Allocates or releases scratchpad bank memory.
    Corresponds to bb_mset instruction (funct7=32).
  }];
  let arguments = (ins I64:$bankId,
                       DefaultValuedAttr<BoolAttr, "true">:$alloc,
                       DefaultValuedAttr<I64Attr, "1">:$row,
                       DefaultValuedAttr<I64Attr, "1">:$col);
  let assemblyFormat = "$bankId attr-dict `:` type($bankId)";
}

def FenceOp : Buckyball_Op<"fence"> {
  let summary = "Memory fence for synchronization";
  let description = [{
    Ensures all previous memory operations complete before proceeding.
    Corresponds to bb_fence instruction (funct7=0).
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Bank SSA operations
//===----------------------------------------------------------------------===//

def BankAllocOp : Buckyball_Op<"bank_alloc"> {
  let summary = "Allocate a virtual on-chip bank resource";
  let description = [{
    Create a virtual bank handle in SSA form.
    - row/col encode bank-array shape used by allocator.
    - The result i64 is a virtual bank handle, not physical id.
  }];
  let arguments = (ins
    DefaultValuedAttr<I64Attr, "1">:$row,
    DefaultValuedAttr<I64Attr, "1">:$col);
  let results = (outs I64:$bank);
  let assemblyFormat = "attr-dict";
}

def BankReleaseOp : Buckyball_Op<"bank_release"> {
  let summary = "Release a virtual bank handle";
  let arguments = (ins I64:$bank);
  let assemblyFormat = "$bank attr-dict `:` type($bank)";
}

def BankMvinOp : Buckyball_Op<"bank_mvin"> {
  let summary = "Load into a virtual bank handle";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$bank, I64:$depth,
                       I64:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat =
      "$input $bank $depth $stride attr-dict `:` type($input) type($bank) type($depth) type($stride)";
}

def BankMvoutOp : Buckyball_Op<"bank_mvout"> {
  let summary = "Store from a virtual bank handle";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$bank, I64:$depth,
                       I64:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat =
      "$output $bank $depth $stride attr-dict `:` type($output) type($bank) type($depth) type($stride)";
}

//===----------------------------------------------------------------------===//
// Mega kernel operations
//===----------------------------------------------------------------------===//

def QuantizeTensorF32ToI8Op : Buckyball_Op<"quantize_tensor_f32_to_i8"> {
  let summary = "Quantize one FP32 model input tensor to INT8";
  let arguments = (ins MemRefRankOf<[F32], [2, 4]>:$input,
                       MemRefRankOf<[I8], [2, 4]>:$output,
                       F32Attr:$scale,
                       BoolAttr:$nchwToNhwc);
  let assemblyFormat = [{$input $output attr-dict `:` type($input) type($output)}];
}

def MegaYieldOp : Buckyball_Op<"mega_yield", [Terminator]> {
  let summary = "Terminate a Buckyball MegaKernel region";
  let assemblyFormat = "attr-dict";
}

def MegaKernelOp : Buckyball_Op<"mega_kernel", [
    SingleBlockImplicitTerminator<"MegaYieldOp">]> {
  let summary = "One cross-layer resident quantized kernel";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output) $body}];
}

def MegaMatmulOp : Buckyball_Op<"mega_matmul"> {
  let summary = "Quantized MatMul stage inside a MegaKernel";
  let description = [{
    One INT8 MatMul stage. Bias is already converted offline to INT32. A
    non-final result is requantized to INT8 and stays inside the MegaKernel.
  }];
  let arguments = (ins MemRefRankOf<[I8], [2]>:$input,
                       MemRefRankOf<[I8], [2]>:$weight,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [2]>:$output,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $weight $bias $scale $lut $output attr-dict `:`
    type($input) type($weight) type($bias) type($scale) type($lut) type($output)}];
}

def MegaConv2dOp : Buckyball_Op<"mega_conv2d"> {
  let summary = "Quantized convolution stage inside a MegaKernel";
  let description = [{
    One resident NHWC INT8 convolution stage. Bias is already converted
    offline to INT32. A non-final result is requantized to INT8 and stays
    inside the MegaKernel.
  }];
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$weight,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [4]>:$output,
                       I64Attr:$kernel,
                       I64Attr:$stride,
                       I64Attr:$padLow,
                       I64Attr:$padHigh,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $weight $bias $scale $lut $output attr-dict `:`
    type($input) type($weight) type($bias) type($scale) type($lut) type($output)}];
}

def MegaConv2dDepthwiseOp : Buckyball_Op<"mega_conv2d_depthwise"> {
  let summary = "Quantized depthwise stage inside a MegaKernel";
  let description = [{
    One complete resident NHWC INT8 depthwise convolution layer with channel
    multiplier one. Bias is converted offline to INT32 and output is
    requantized to INT8 before leaving the mega kernel.
  }];
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$weight,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [4]>:$output,
                       I64Attr:$kernel,
                       I64Attr:$stride,
                       I64Attr:$padLow,
                       I64Attr:$padHigh,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $weight $bias $scale $lut $output attr-dict `:`
    type($input) type($weight) type($bias) type($scale) type($lut) type($output)}];
}

def MegaMaxPool2dOp : Buckyball_Op<"mega_max_pool2d"> {
  let summary = "INT8 MaxPool2D stage inside a MegaKernel";
  let description = [{
    Max pooling preserves the INT8 quantization scale. An intermediate output
    is NHWC; a final output is NCHW so a following reshape preserves PyTorch
    flatten order without a transpose operation.
  }];
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$output,
                       I64Attr:$kernel,
                       I64Attr:$stride,
                       I64Attr:$padding,
                       BoolAttr:$finalOutput);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def MegaGlobalAvgPoolOp : Buckyball_Op<"mega_global_avg_pool"> {
  let summary = "Global-average an INT8 NHWC tensor inside a MegaKernel";
  let description = [{
    This is the explicit X-memory boundary used by an SE branch. The result is
    requantized to INT8 before the following pointwise stage.
  }];
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$inputScale,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def MegaInt8MulOp : Buckyball_Op<"mega_int8_mul"> {
  let summary = "Scaled INT8 broadcast multiply inside a MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$lhs,
                       MemRefRankOf<[I8], [4]>:$rhs,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$lhsScale,
                       F32Attr:$rhsScale,
                       F32Attr:$outputScale,
                       I64Attr:$activation);
  let assemblyFormat = [{$lhs $rhs $output attr-dict `:`
    type($lhs) type($rhs) type($output)}];
}

def MegaInt8AddOp : Buckyball_Op<"mega_int8_add"> {
  let summary = "Scaled INT8 residual add inside a MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$lhs,
                       MemRefRankOf<[I8], [4]>:$rhs,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$lhsScale,
                       F32Attr:$rhsScale,
                       F32Attr:$outputScale,
                       I64Attr:$activation);
  let assemblyFormat = [{$lhs $rhs $output attr-dict `:`
    type($lhs) type($rhs) type($output)}];
}

//===----------------------------------------------------------------------===//
// LLVM intrinsic operations
//===----------------------------------------------------------------------===//

class Buckyball_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Buckyball_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_bb_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

class Buckyball_CustomIntrOpBase<string mnemonic, string enumName,
                                 list<int> immArgPositions,
                                 list<string> immArgAttrNames,
                                 list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Buckyball_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/enumName,
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0,
                  /*bit requiresAccessGroup=*/0,
                  /*bit requiresAliasAnalysis=*/0,
                  /*bit requiresFastmath=*/0,
                  /*bit requiresArgAndResultAttrs=*/0,
                  /*bit requiresOpBundles=*/0,
                  /*list<int> immArgPositions=*/immArgPositions,
                  /*list<string> immArgAttrNames=*/immArgAttrNames>;

def MsetIntrOp : Buckyball_IntrOpBase<"mset">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def FenceIntrOp : Buckyball_IntrOpBase<"fence">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def MvinIntrOp : Buckyball_IntrOpBase<"mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def MvoutIntrOp : Buckyball_IntrOpBase<"mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def RushBMvinOp : Buckyball_Op<"rushb_mvin"> {
  let summary = "rushB host DMA load";
  let arguments = (ins LLVM_Type:$rs1, LLVM_Type:$packedXs2,
                       LLVM_Type:$hostPtr);
  let assemblyFormat = "$rs1 `,` $packedXs2 `,` $hostPtr attr-dict `:` "
                       "type($rs1) `,` type($packedXs2) `,` type($hostPtr)";
}

def RushBMvoutOp : Buckyball_Op<"rushb_mvout"> {
  let summary = "rushB host DMA store";
  let arguments = (ins LLVM_Type:$rs1, LLVM_Type:$packedXs2,
                       LLVM_Type:$hostPtr);
  let assemblyFormat = "$rs1 `,` $packedXs2 `,` $hostPtr attr-dict `:` "
                       "type($rs1) `,` type($packedXs2) `,` type($hostPtr)";
}

def CustomIntrOp
    : Buckyball_CustomIntrOpBase<"custom", "riscv_buckyball_custom",
                                 [2], ["funct7"]>,
      Arguments<(ins LLVM_Type:$rs1, LLVM_Type:$rs2, I32Attr:$funct7)> {
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict `:` type($rs1) `,` type($rs2)";
}

//===----------------------------------------------------------------------===//
// Generic instruction and debug operations
//===----------------------------------------------------------------------===//

def InstOp : Buckyball_Op<"inst"> {
  let summary = "Generic Buckyball custom instruction carrier";
  let description = [{
    Emits a Buckyball custom instruction with explicit rs1, rs2 and funct7.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2,
                       ConfinedAttr<I32Attr,
                                    [IntMinValue<0>, IntMaxValue<127>]>:$funct7);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict `:` type($rs1)";
}

def PrintMemRefOp : Buckyball_Op<"print_memref"> {
  let summary = "Print memref contents for debugging";
  let description = [{
    Print the contents of a memref for debugging purposes.
  }];
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintScalarOp : Buckyball_Op<"print_scalar"> {
  let summary = "Print scalar value for debugging";
  let description = [{
    Print a scalar value for debugging purposes.
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // BUCKYBALL_OPS
