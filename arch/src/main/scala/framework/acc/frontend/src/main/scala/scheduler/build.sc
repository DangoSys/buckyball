import mill._
import $file.src.main.scala.bat.{build => bat_build}
import $file.src.main.scala.rob.global.{build => rob_global_build}
import $file.src.main.scala.scoreboard.{build => scoreboard_build}
import $file.^.^.^.^.^.^.^.^.^.^.settings
val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.frontend.scheduler.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Scheduler.scala"))
  ) ++
  bat_build.sources() ++
  rob_global_build.sources() ++
  scoreboard_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  bat_build.emitSources() ++
  rob_global_build.emitSources() ++
  scoreboard_build.emitSources()
)
