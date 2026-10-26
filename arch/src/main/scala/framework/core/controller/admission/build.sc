import mill._
import $file.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.core.controller.admission.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/CommandAdmission.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/DirectAdmission.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/TaskRouter.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
