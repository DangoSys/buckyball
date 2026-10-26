import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.thread.ant.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/AntParam.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Ant.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Decode.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/Executor.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/LocalStore.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
