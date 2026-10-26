import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.ip.bus.axi4.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Decoder.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Interconnect.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Interface.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
