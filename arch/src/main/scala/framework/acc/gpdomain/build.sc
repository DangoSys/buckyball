import mill._
import $file.src.main.scala.execution.{build => execution_build}
import $file.src.main.scala.loader.{build => loader_build}
import $file.src.main.scala.memory.{build => memory_build}
import $file.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.gpdomain.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/GpDomain.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/GpDomainParam.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/DISA.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/Decoder.scala"))
  ) ++
  execution_build.sources() ++
  loader_build.sources() ++
  memory_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  execution_build.emitSources() ++
  loader_build.emitSources() ++
  memory_build.emitSources()
)
