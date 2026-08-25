//===-- TraceBall.td - Trace Ball operations ----------------------------===//

#ifndef TRACE_BALL_OPS
#define TRACE_BALL_OPS

include "Buckyball.td"

def BdbCounterOp : Buckyball_Op<"bdb_counter"> {
  let summary = "Start, stop, or read a Buckyball debug counter";
  let description = [{
    Encodes BDB_COUNTER rs2 as subcommand[3:0], counterId[7:4], and
    payload[63:8]. rs1 is always zero.
  }];
  let arguments = (ins I64:$subcommand, I64:$counterId, I64:$payload);
  let assemblyFormat = "$subcommand `,` $counterId `,` $payload attr-dict `:` type($subcommand)";
}

#endif // TRACE_BALL_OPS
