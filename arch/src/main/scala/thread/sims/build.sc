import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Elaborate.scala")),
    PathRef(millSourcePath / os.RelPath("ThreadEnvironment.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
