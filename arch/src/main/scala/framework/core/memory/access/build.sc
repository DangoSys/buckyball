import mill._
import $file.src.main.scala.cache.{build => cache_build}
import $file.src.main.scala.mmu.{build => mmu_build}
import $file.src.main.scala.preflight.{build => preflight_build}
import $file.^.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.core.memory.access.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/PhysicalAccess.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/VirtualAccess.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  ) ++
  cache_build.sources() ++
  mmu_build.sources() ++
  preflight_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  cache_build.emitSources() ++
  mmu_build.emitSources() ++
  preflight_build.emitSources()
)
