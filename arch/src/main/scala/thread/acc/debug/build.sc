import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("BankCheckDPI.scala")),
    PathRef(millSourcePath / os.RelPath("Debug.scala")),
    PathRef(millSourcePath / os.RelPath("trace/AntTraceDPI.scala")),
    PathRef(millSourcePath / os.RelPath("trace/ITraceDPI.scala")),
    PathRef(millSourcePath / os.RelPath("trace/MTraceDPI.scala")),
    PathRef(millSourcePath / os.RelPath("trace/PMCTraceDPI.scala")),
    PathRef(millSourcePath / os.RelPath("trace/Trace.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
