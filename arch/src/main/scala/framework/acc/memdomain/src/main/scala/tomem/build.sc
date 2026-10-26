import mill._
import $file.src.main.scala.toddr.{build => toddr_build}
import $file.src.main.scala.tol1.{build => tol1_build}
import $file.^.^.^.^.^.^.^.^.^.^.settings

val ipRoot = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.acc.memdomain.tomem.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/ToMem.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  ) ++
  toddr_build.sources() ++
  tol1_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  toddr_build.emitSources() ++
  tol1_build.emitSources()
)
