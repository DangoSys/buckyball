import mill._
import $file.admission.{build => admission_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Controller.scala")),
    PathRef(millSourcePath / os.RelPath("configs/ControllerParam.scala")),
    PathRef(millSourcePath / os.RelPath("task/Control.scala")),
    PathRef(millSourcePath / os.RelPath("task/Service.scala"))
  ) ++
  admission_build.sources()
)

def emitSources = T.sources(
  admission_build.emitSources()
)
