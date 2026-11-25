import mill._
import $file.src.main.scala.router.{build => router_build}
import $file.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.tlp.tbus.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/SharedNetwork.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  ) ++
    router_build.sources()
)

def emitSources = T.sources(
  Seq.empty[PathRef] ++
    router_build.emitSources()
)
