import mill._
import $file.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.gpdomain.execution.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Execution.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/IBuf.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/arithmetic/FALU.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/arithmetic/IALU.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/scalar/ScalarExecution.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/scalar/ScalarFloating.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/scalar/ScalarRF.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/vector/Operands.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/vector/VRF.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/vector/VectorThread.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
