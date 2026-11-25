import mill._
import $file.^.^.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.custom.gpdomain.memory.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/BankPort.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/OperatorMemory.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Memory.scala")),
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
