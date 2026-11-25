//===- layernormOps.td - LayerNormBall operations ------------*- tablegen -*-===//
//
// layernorm (input, params) -> output: the layer normalization of `rows` FP32
// rows of length `w` (16 <= w <= 2048; ceil(w/4) bank rows each), one
// LAYERNORM instruction (workloads/isa/layernorm.h; the reference is
// emu/src/layernorm.rs). params holds gamma, beta (ceil(w/4) bank rows each)
// and eps (lane 0 of one bank row). Input, params and output are distinct
// banks of one 128-bit wide group each and start at bank row 0: the bank-row
// fields of the instruction are 0.
//
//===----------------------------------------------------------------------===//

#ifndef LAYERNORM_OPS
#define LAYERNORM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def LayerNormBall_Dialect : Dialect {
  let name = "layernorm_ball";
  let cppNamespace = "::buddy::layernorm_ball";
  let summary = "The operations of LayerNormBall";
}

class LayerNormBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<LayerNormBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankLayerNormOp : LayerNormBall_Op<"bank_layernorm",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "layernorm on virtual banks";
  let arguments = (ins I64:$input, I64:$params, I64:$output, I64Attr:$rows,
                       I64Attr:$w, BoolAttr:$rms);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `,` $params `->` $output prop-dict attr-dict";
}

def LayerNormOp : LayerNormBall_Op<"layernorm",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "layernorm on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$params, I64Attr:$output,
                   I64Attr:$rows, I64Attr:$w, BoolAttr:$rms);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // LAYERNORM_OPS
