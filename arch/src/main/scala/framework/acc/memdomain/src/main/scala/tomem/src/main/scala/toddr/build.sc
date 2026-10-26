import mill._
import $file.^.^.^.^.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.toddr.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/DmaPort.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/DmaStatus.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ReadDma.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/ToDDR.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/WriteDma.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
