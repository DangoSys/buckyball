import mill._

// Instruction-level parallelism: what lets a CPU overlap instructions
// (pipelining, bypassing, branch prediction, multi-issue, out-of-order). A CPU
// runs without it, one instruction at a time.
def sources = T.sources(
  Seq.empty[PathRef]
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
