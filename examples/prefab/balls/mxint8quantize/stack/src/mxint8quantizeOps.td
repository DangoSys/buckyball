//===- mxint8quantizeOps.td - Mxint8quantizeBall operations -*- tablegen -*-===//
//
// quant_mxint8 (input) -> output: quantizes `rows` rows of `k` FP32 (k a
// multiple of 32) to packed MXINT8 rows, one MXINT8QUANT instruction. Input
// row r holds its k finite FP32 at bank row r * k / 4. Output row r starts at
// bank row r * (k / 16 + ceil(k / 512)): the k INT8 codes, then the k / 32
// exponent bytes, zero-padded to whole bank rows. Every 32 values share the
// exponent e = floor(log2(max |x|)) - 6, at least -127 (0 for an all-zero
// block), stored as e + 127; a code is clamp(round_half_even(x / 2^e), -127,
// 127), as quantize_mxint8 of stack/compiler/frontend/quant.py.
// Input and output are distinct banks of one 128-bit wide group each.
//
//===----------------------------------------------------------------------===//

#ifndef MXINT8QUANTIZE_OPS
#define MXINT8QUANTIZE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8quantizeBall_Dialect : Dialect {
  let name = "mxint8quantize_ball";
  let cppNamespace = "::buddy::mxint8quantize_ball";
  let summary = "The operations of Mxint8quantizeBall";
}

class Mxint8quantizeBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8quantizeBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankQuantMxint8Op : Mxint8quantizeBall_Op<"bank_quant_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "quant_mxint8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$rows, I64Attr:$k);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def QuantMxint8Op : Mxint8quantizeBall_Op<"quant_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "quant_mxint8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$rows,
                   I64Attr:$k);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXINT8QUANTIZE_OPS
