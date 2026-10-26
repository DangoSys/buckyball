import mill._
import $file.^.^.^.^.^.^.^.^.^.^.^.^.^.^.^.settings
val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.frontend.scheduler.rob.global.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/DependencyResolver.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/GlobalROB.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Rob.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/RobParams.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
