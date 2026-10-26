import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("DivRem.scala")),
    PathRef(millSourcePath / os.RelPath("Fma.scala")),
    PathRef(millSourcePath / os.RelPath("IntToFloat.scala")),
    PathRef(millSourcePath / os.RelPath("Multiply.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
