//===- mxint8quantOps.td - Mxint8quantBall operations ------*- tablegen -*-===//
//
// quant_mxint8 (input) -> output: quantizes `rows` FP32 rows of length `k` (a
// multiple of 32; k/4 bank rows each) to packed MXINT8 rows (k/16 +
// ceil(k/512) bank rows each: the k INT8 codes, the k/32 exponent bytes
// e + 127, zero padding), one MXINT8QUANT instruction. Input and output start
// at bank row 0: the bank-row fields of the instruction are 0. Every 32 values share e = floor(log2(max |x|)) - 6 (at least
// -127; 0 for an all-zero block) and code clamp(round_half_even(x / 2^e),
// -127, 127): quantize_mxint8 of stack/compiler/frontend/quant.py.
// Input and output are distinct banks of one 128-bit wide group each.
//
//===----------------------------------------------------------------------===//

#ifndef MXINT8QUANT_OPS
#define MXINT8QUANT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8quantBall_Dialect : Dialect {
  let name = "mxint8quant_ball";
  let cppNamespace = "::buddy::mxint8quant_ball";
  let summary = "The operations of Mxint8quantBall";
}

class Mxint8quantBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8quantBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankQuantMxint8Op : Mxint8quantBall_Op<"bank_quant_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "quant_mxint8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$rows, I64Attr:$k);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def QuantMxint8Op : Mxint8quantBall_Op<"quant_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "quant_mxint8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$rows,
                   I64Attr:$k);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXINT8QUANT_OPS
