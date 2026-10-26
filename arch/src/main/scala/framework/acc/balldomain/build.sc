import mill._
import $file.src.main.scala.ball.{build => ball_build}
import $file.src.main.scala.bbus.{build => bbus_build}
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/BallDomain.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/mmio_bank/MmioBank.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/sub_rob/SubROB.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/sub_rob/Expansion.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/sub_rob/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/BallDomainParam.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/DISA.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/decoder/Decoder.scala"))
  ) ++
  ball_build.sources() ++
  bbus_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  ball_build.emitSources() ++
  bbus_build.emitSources()
)
