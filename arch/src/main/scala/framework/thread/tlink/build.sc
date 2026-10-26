import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("basethread.scala")),
    PathRef(millSourcePath / os.RelPath("tlink.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
