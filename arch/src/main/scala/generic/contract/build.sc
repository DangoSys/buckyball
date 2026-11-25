import mill._

// The vocabulary shared across abstractions: interface bundles and their parameters.
def sources = T.sources(millSourcePath)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
