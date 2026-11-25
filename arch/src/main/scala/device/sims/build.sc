import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/GlobalConfig.scala")),
    PathRef(millSourcePath / os.RelPath("soc/SimSoc.scala")),
    PathRef(millSourcePath / os.RelPath("verilator/BBsimDRAM.scala")),
    PathRef(millSourcePath / os.RelPath("verilator/BdbClkDPI.scala")),
    PathRef(millSourcePath / os.RelPath("verilator/SystemHarness.scala")),
    PathRef(millSourcePath / os.RelPath("verilator/TargetConfig.scala")),
    PathRef(millSourcePath / os.RelPath("p2e/P2ETop.scala")),
    PathRef(millSourcePath / os.RelPath("p2e/TargetConfig.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
