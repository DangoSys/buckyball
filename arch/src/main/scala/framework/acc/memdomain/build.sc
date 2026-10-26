import mill._
import $file.src.main.scala.bank_router.{build => bank_router_build}
import $file.src.main.scala.prm.{build => prm_build}
import $file.src.main.scala.tomem.{build => tomem_build}
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/MemDomain.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MemoryControl.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/MemDomainParam.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ctrl/Ctrl.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ctrl/configer/Configer.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ctrl/decoder/DISA.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ctrl/decoder/Decoder.scala"))
  ) ++
  bank_router_build.sources() ++
  prm_build.sources() ++
  tomem_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  bank_router_build.emitSources() ++
  prm_build.emitSources() ++
  tomem_build.emitSources()
)
