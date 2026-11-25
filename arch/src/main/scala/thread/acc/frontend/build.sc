import mill._
import $file.src.main.scala.boot.{build => boot_build}
import $file.src.main.scala.scheduler.{build => scheduler_build}
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.thread.acc.frontend.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Frontend.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Ingress.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/DISA.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/GlobalDecoder.scala"))
  ) ++
  boot_build.sources() ++
  scheduler_build.sources()
)

def emitSources = T.sources(
  Seq.empty[PathRef] ++
  boot_build.emitSources() ++
  scheduler_build.emitSources()
)
