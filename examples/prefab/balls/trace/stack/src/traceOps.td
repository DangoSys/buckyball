//===- traceOps.td - TraceBall operations ------------------*- tablegen -*-===//
//
// The BDB counter touches no bank: its operations exist only on physical
// banks, with no sources and destination 0, and rs1 is zero. `counter` (0 to
// 15) selects the counter. counter_start starts it with the i64 tag;
// counter_read traces its current value; counter_stop traces the elapsed
// cycles and stops it.
//
//===----------------------------------------------------------------------===//

#ifndef TRACE_OPS
#define TRACE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def TraceBall_Dialect : Dialect {
  let name = "trace_ball";
  let cppNamespace = "::buddy::trace_ball";
  let summary = "The operations of TraceBall";
}

class TraceBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<TraceBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def CounterStartOp : TraceBall_Op<"counter_start",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "Start a BDB counter with a tag";
  let arguments = (ins I64:$tag, I64Attr:$counter);
  let assemblyFormat = "$tag prop-dict attr-dict";
}

def CounterStopOp : TraceBall_Op<"counter_stop",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "Trace the elapsed cycles of a BDB counter and stop it";
  let arguments = (ins I64Attr:$counter);
  let assemblyFormat = "prop-dict attr-dict";
}

def CounterReadOp : TraceBall_Op<"counter_read",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "Trace the current value of a BDB counter";
  let arguments = (ins I64Attr:$counter);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // TRACE_OPS
