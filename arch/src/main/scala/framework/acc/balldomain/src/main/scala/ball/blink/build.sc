import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("baseball.scala")),
    PathRef(millSourcePath / os.RelPath("blink.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
