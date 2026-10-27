//===- toint8Ops.td - ToInt8Ball operations ----------------*- tablegen -*-===//
//
// fp32_to_int8: the input bank holds `rows` rows of four f32, a multiple of
// four; the output receives rows / 4 rows of sixteen i8,
// clamp(round_half_even(x * multiplier), -128, 127). `multiplier` is f32.
// int32_to_int8: `values` holds `rows` rows of sixteen i32 (four lines each),
// `scales` the sixteen f32 scales of the columns (four lines); the output
// receives `rows` rows of sixteen i8, clamp(round_half_even(x * scale), -128,
// 127), with `relu` at least zero.
// Both work on every group: group g is channel block g, and the banks have the
// same number of groups.
//
//===----------------------------------------------------------------------===//

#ifndef TOINT8_OPS
#define TOINT8_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def ToInt8Ball_Dialect : Dialect {
  let name = "toint8_ball";
  let cppNamespace = "::buddy::toint8_ball";
  let summary = "The operations of ToInt8Ball";
}

class ToInt8Ball_Op<string mnemonic, list<Trait> traits = []>
    : Op<ToInt8Ball_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankFp32ToInt8Op : ToInt8Ball_Op<"bank_fp32_to_int8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "fp32_to_int8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, F32:$multiplier,
                       I64Attr:$rows);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output `multiplier` `(` $multiplier `)` "
                       "prop-dict attr-dict";
}

def Fp32ToInt8Op : ToInt8Ball_Op<"fp32_to_int8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "fp32_to_int8 on physical banks";
  let arguments = (ins F32:$multiplier, I64Attr:$input, I64Attr:$output,
                       I64Attr:$rows);
  let assemblyFormat = "`multiplier` `(` $multiplier `)` prop-dict attr-dict";
}

def BankInt32ToInt8Op : ToInt8Ball_Op<"bank_int32_to_int8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "int32_to_int8 on virtual banks";
  let arguments = (ins I64:$values, I64:$scales, I64:$output, I64Attr:$rows,
                       UnitAttr:$relu);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$values `,` $scales `->` $output prop-dict attr-dict";
}

def Int32ToInt8Op : ToInt8Ball_Op<"int32_to_int8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "int32_to_int8 on physical banks";
  let arguments = (ins I64Attr:$values, I64Attr:$scales, I64Attr:$output,
                       I64Attr:$rows, UnitAttr:$relu);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // TOINT8_OPS
