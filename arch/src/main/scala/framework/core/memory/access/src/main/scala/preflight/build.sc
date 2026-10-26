import mill._
import $file.^.^.^.^.^.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.core.memory.access.preflight.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Permission.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/PermissionCheck.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Preflight.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Preparation.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/PreparedMap.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Segmenter.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Config.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
