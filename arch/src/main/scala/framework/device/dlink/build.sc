import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("basedevice.scala")),
    PathRef(millSourcePath / os.RelPath("dlink.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
