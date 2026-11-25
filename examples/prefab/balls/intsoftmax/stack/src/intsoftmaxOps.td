#ifndef INTSOFTMAX_OPS
#define INTSOFTMAX_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def IntSoftmaxBall_Dialect : Dialect {
  let name = "intsoftmax_ball";
  let cppNamespace = "::buddy::intsoftmax_ball";
  let summary = "Integer Q8/Q15 attention softmax";
}
class IntSoftmaxBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntSoftmaxBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankSoftmaxOp : IntSoftmaxBall_Op<"bank_softmax",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let arguments = (ins I64:$input, I64:$table, I64:$output,
                       I64:$rows, I64:$columns);
  let results = (outs I64:$outputState);
  let assemblyFormat = "$input `,` $table `->` $output `rows` $rows `columns` $columns attr-dict";
}

def SoftmaxOp : IntSoftmaxBall_Op<"softmax",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let arguments = (ins I64Attr:$input, I64Attr:$table, I64Attr:$output,
                       I64:$rows, I64:$columns);
  let assemblyFormat = "$rows `,` $columns prop-dict attr-dict";
}
#endif
