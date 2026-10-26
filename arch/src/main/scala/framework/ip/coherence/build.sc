import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.ip.memory.coherence.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/CoherenceParams.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Coherence.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Directory.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Interface.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Mshr.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
