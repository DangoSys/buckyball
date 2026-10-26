import mill._
import $file.^.^.^.^.^.^.^.^.^.^.^.^.^.^.settings
val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.frontend.scheduler.scoreboard.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/BankAccessInfo.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/BankScoreboard.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
