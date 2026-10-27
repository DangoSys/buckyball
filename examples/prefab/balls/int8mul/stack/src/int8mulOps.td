//===- int8mulOps.td - Int8MulBall operations --------------*- tablegen -*-===//
//
// gate_mul_i8: `gate` holds the gate, one row of sixteen i8; `input` holds
// `rows` rows of sixteen i8. The output receives
// clamp(round_half_even(gate * x * ratio), -128, 127) per column, in every
// group of the banks; the three banks have the same number of groups. `ratio`
// is f32, finite and positive.
//
//===----------------------------------------------------------------------===//

#ifndef INT8MUL_OPS
#define INT8MUL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def Int8MulBall_Dialect : Dialect {
  let name = "int8mul_ball";
  let cppNamespace = "::buddy::int8mul_ball";
  let summary = "The operations of Int8MulBall";
}

class Int8MulBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Int8MulBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankGateMulI8Op : Int8MulBall_Op<"bank_gate_mul_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "gate_mul_i8 on virtual banks";
  let arguments = (ins I64:$gate, I64:$input, I64:$output, F32:$ratio,
                       I64Attr:$rows);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$gate `,` $input `->` $output `ratio` `(` $ratio `)` "
                       "prop-dict attr-dict";
}

def GateMulI8Op : Int8MulBall_Op<"gate_mul_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "gate_mul_i8 on physical banks";
  let arguments = (ins F32:$ratio, I64Attr:$gate, I64Attr:$input,
                       I64Attr:$output, I64Attr:$rows);
  let assemblyFormat = "`ratio` `(` $ratio `)` prop-dict attr-dict";
}

#endif // INT8MUL_OPS
