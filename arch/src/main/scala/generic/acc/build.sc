import mill._

// The accelerator vocabulary a hart's Balls and its threads share: the domain
// parameters, the Ball link, the bank layout, routing and access, the memory
// trace. Specialization (custom) and thread-level parallelism (tlp) both use it.
def sources = T.sources(millSourcePath)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
