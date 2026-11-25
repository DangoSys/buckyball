import mill._
import $file.^.^.^.^.^.settings
import $file.access.{build => access_build}
import $file.home.{build => home_build}

val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.hart.memory.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Memory.scala")),
    PathRef(millSourcePath / os.RelPath("Requester.scala")),
    PathRef(millSourcePath / os.RelPath("DmaRequester.scala")),
    PathRef(millSourcePath / os.RelPath("MemorySystem.scala")),
    PathRef(millSourcePath / os.RelPath("MemCtrl.scala")),
    PathRef(millSourcePath / os.RelPath("configs/MemoryParam.scala")),
    PathRef(millSourcePath / os.RelPath("uncached/Uncached.scala")),
    PathRef(millSourcePath / os.RelPath("uncached/configs/UncachedParam.scala"))
  ) ++
  access_build.sources() ++
  home_build.sources()
)

def emitSources = T.sources(
  Seq(PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))) ++ access_build.emitSources() ++
  home_build.emitSources()
)
