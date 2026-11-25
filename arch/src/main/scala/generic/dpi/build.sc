import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("DpiGuard.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
