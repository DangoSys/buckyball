import mill._
import $file.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.core.execution.instruction.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Instruction.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
