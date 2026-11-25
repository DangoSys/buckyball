import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("basehart.scala")),
    PathRef(millSourcePath / os.RelPath("hlink.scala")),
    PathRef(millSourcePath / os.RelPath("ServiceData.scala")),
    PathRef(millSourcePath / os.RelPath("configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
