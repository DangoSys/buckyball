//===- mxfp2intOps.td - Mxfp2IntBall operations ------------*- tablegen -*-===//
//
// mxfp4_i8 (data, scales) -> output: dequantizes `rows` MXFP4 blocks of 32
// elements to i8, one MXFP2INT instruction. Banks are one group of sixteen
// bytes per row:
//   data: block b in row b, element e in byte e / 2, low nibble first; the
//     element value is raw - 8.
//   scales: the E8M0 scale of block b in byte b % 16 of row b / 16.
//   output: elements 0..15 of block b in row 2b, 16..31 in row 2b + 1,
//     saturate(value * 2^(scale - 127)), the right shift rounding toward minus
//     infinity.
// `rows` is the number of blocks, 1..half the bank depth. The Ball must be
// configured for MXFP4.
//
//===----------------------------------------------------------------------===//

#ifndef MXFP2INT_OPS
#define MXFP2INT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def Mxfp2IntBall_Dialect : Dialect {
  let name = "mxfp2int_ball";
  let cppNamespace = "::buddy::mxfp2int_ball";
  let summary = "The operations of Mxfp2IntBall";
}

class Mxfp2IntBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxfp2IntBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMxfp4I8Op : Mxfp2IntBall_Op<"bank_mxfp4_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "mxfp4_i8 on virtual banks";
  let arguments = (ins I64:$data, I64:$scales, I64:$output, I64Attr:$rows);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$data `,` $scales `->` $output prop-dict attr-dict";
}

def Mxfp4I8Op : Mxfp2IntBall_Op<"mxfp4_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "mxfp4_i8 on physical banks";
  let arguments = (ins I64Attr:$data, I64Attr:$scales, I64Attr:$output,
                       I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXFP2INT_OPS
