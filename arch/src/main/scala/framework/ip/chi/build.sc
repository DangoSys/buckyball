import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.ip.bus.chi.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Channel.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/CoherenceState.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/DirectedSnoop.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Fabric.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Flit.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/HomeMapping.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/RequestRetry.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/RequesterPort.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Rx.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Tx.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/hnf/ChiHome.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/rnf/AxiCache.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/rnf/BankedChiCache.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/rnf/CachePort.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/rnf/ChiCache.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/snf/MemoryNode.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/snf/SramEndpoint.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
