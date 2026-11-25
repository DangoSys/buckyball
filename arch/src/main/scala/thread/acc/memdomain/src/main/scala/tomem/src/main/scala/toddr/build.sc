import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/DmaPort.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
