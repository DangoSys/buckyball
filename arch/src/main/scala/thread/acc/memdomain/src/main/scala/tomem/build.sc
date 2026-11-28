import mill._
import $file.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.thread.acc.memdomain.tomem.Emit")
}
import $file.src.main.scala.toddr.{build => toddr_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/ToMem.scala")),
  ) ++
  toddr_build.sources()
)

def emitSources = T.sources(
  toddr_build.emitSources()
)
