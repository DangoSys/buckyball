import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Arbiter.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
