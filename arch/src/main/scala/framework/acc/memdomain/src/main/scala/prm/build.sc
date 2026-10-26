import mill._
import $file.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.prm.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Bank.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MvoverIO.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/PRM.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
