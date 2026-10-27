//===- mxquantOps.td - MxquantBall operations --------------*- tablegen -*-===//
//
// quant_mxfp8 (input) -> output: quantizes `count` FP32 (a multiple of 32) to
// block32 MXFP8, one MXQUANT instruction. Input holds the `count` finite FP32
// from byte 0; output receives the `count` E4M3 codes from byte 0, then one
// E8M0 scale per block of 32 from byte `count`. The scale is the block
// exponent: the largest magnitude's exponent minus 8, at least -127, biased by
// 127 (0 for an all-zero block); a code rounds to nearest even, saturates at
// magnitude code 126 and keeps the sign of zero. Nothing else of the output
// is written.
// Input and output are distinct banks of one 128-bit wide bank each.
//
//===----------------------------------------------------------------------===//

#ifndef MXQUANT_OPS
#define MXQUANT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def MxquantBall_Dialect : Dialect {
  let name = "mxquant_ball";
  let cppNamespace = "::buddy::mxquant_ball";
  let summary = "The operations of MxquantBall";
}

class MxquantBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<MxquantBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankQuantMxfp8Op : MxquantBall_Op<"bank_quant_mxfp8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "quant_mxfp8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$count);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def QuantMxfp8Op : MxquantBall_Op<"quant_mxfp8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "quant_mxfp8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$count);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXQUANT_OPS
