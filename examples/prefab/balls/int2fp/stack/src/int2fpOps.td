//===- int2fpOps.td - Int2FpBall operations ----------------*- tablegen -*-===//
//
// int32_to_fp32: `values` holds `rows` rows of sixteen i32 (four bank rows
// each), `scales` the sixteen f32 scales of the columns (four bank rows); the
// output receives the f32 value * scale of every element, with `relu`
// negative results replaced by zero.
//
//===----------------------------------------------------------------------===//

#ifndef INT2FP_OPS
#define INT2FP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def Int2FpBall_Dialect : Dialect {
  let name = "int2fp_ball";
  let cppNamespace = "::buddy::int2fp_ball";
  let summary = "The operations of Int2FpBall";
}

class Int2FpBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Int2FpBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankInt32ToFp32Op : Int2FpBall_Op<"bank_int32_to_fp32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "int32_to_fp32 on virtual banks";
  let arguments = (ins I64:$values, I64:$scales, I64:$output, I64Attr:$rows,
                       UnitAttr:$relu);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$values `,` $scales `->` $output prop-dict attr-dict";
}

def Int32ToFp32Op : Int2FpBall_Op<"int32_to_fp32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "int32_to_fp32 on physical banks";
  let arguments = (ins I64Attr:$values, I64Attr:$scales, I64Attr:$output,
                       I64Attr:$rows, UnitAttr:$relu);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // INT2FP_OPS
