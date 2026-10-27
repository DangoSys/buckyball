//===- int8addOps.td - Int8AddBall operations --------------*- tablegen -*-===//
//
// add_i8: c receives `rows` rows of sixteen i8,
// round_half_even(a * lhs_ratio + b * rhs_ratio) saturated to i8, and with
// `relu` clamped at zero. lhs_ratio and rhs_ratio are f32, both finite and
// positive.
//
//===----------------------------------------------------------------------===//

#ifndef INT8ADD_OPS
#define INT8ADD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def Int8AddBall_Dialect : Dialect {
  let name = "int8add_ball";
  let cppNamespace = "::buddy::int8add_ball";
  let summary = "The operations of Int8AddBall";
}

class Int8AddBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Int8AddBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankAddI8Op : Int8AddBall_Op<"bank_add_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "add_i8 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, F32:$lhs_ratio,
                       F32:$rhs_ratio, I64Attr:$rows, UnitAttr:$relu);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c `ratios` `(` $lhs_ratio `,` "
                       "$rhs_ratio `)` prop-dict attr-dict";
}

def AddI8Op : Int8AddBall_Op<"add_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "add_i8 on physical banks";
  let arguments = (ins F32:$lhs_ratio, F32:$rhs_ratio, I64Attr:$a,
                       I64Attr:$b, I64Attr:$c, I64Attr:$rows, UnitAttr:$relu);
  let assemblyFormat = "`ratios` `(` $lhs_ratio `,` $rhs_ratio `)` prop-dict "
                       "attr-dict";
}

#endif // INT8ADD_OPS
