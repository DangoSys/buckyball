import mill._
import $file.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.bank_router.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/BankRouter.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/blt/AccPipe.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/blt/BLT.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
