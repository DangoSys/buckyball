import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("basecore.scala")),
    PathRef(millSourcePath / os.RelPath("clink.scala")),
    PathRef(millSourcePath / os.RelPath("configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
