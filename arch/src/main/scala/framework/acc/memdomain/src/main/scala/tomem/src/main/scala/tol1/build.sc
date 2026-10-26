import mill._
import $file.^.^.^.^.^.^.^.^.^.^.^.^.^.^.settings
val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.tol1.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/ToL1.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
