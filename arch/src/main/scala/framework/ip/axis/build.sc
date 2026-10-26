import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.ip.bus.axi.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Interface.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Master.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/PacketArbiter.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Slave.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
