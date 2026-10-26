import mill._

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Controller.scala")),
    PathRef(millSourcePath / os.RelPath("bootrom/BootRom.scala")),
    PathRef(millSourcePath / os.RelPath("bootrom/configs/Parameters.scala")),
    PathRef(millSourcePath / os.RelPath("clint/Clint.scala")),
    PathRef(millSourcePath / os.RelPath("clint/configs/Parameters.scala")),
    PathRef(millSourcePath / os.RelPath("plic/Plic.scala")),
    PathRef(millSourcePath / os.RelPath("plic/configs/Parameters.scala")),
    PathRef(millSourcePath / os.RelPath("scu/SCU.scala")),
    PathRef(millSourcePath / os.RelPath("scu/SystemControl.scala")),
    PathRef(millSourcePath / os.RelPath("scu/configs/Parameters.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
