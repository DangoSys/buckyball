import mill._
import $file.^.^.^.^.^.settings
import $file.^.^.^.^.^.thirdparty.{build => thirdparty_build}

val ipRoot: os.Path = millSourcePath
val generic = millSourcePath / os.up / os.up / "generic"

def implementation = T.sources(
  os.walk(millSourcePath / "src" / "main" / "scala").filter(path =>
    os.isFile(path) && path.ext == "scala" && path.last != "Emit.scala"
  ).map(PathRef(_))
)

object library extends settings.FrameworkModule {
  override def moduleRoot = ipRoot
  override def moduleDeps = Seq(thirdparty_build.protoJava)
  override def sources = T.sources(implementation() ++ Seq(
    PathRef(generic / "axis" / "src" / "main" / "scala" / "Interface.scala"),
    PathRef(generic / "bank" / "src" / "main" / "scala" / "Bank.scala"),
    PathRef(generic / "bank" / "src" / "main" / "scala" / "BankSet.scala"),
    PathRef(generic / "queue" / "Queue.scala"),
    PathRef(generic / "arbiter" / "Arbiter.scala"),
    PathRef(generic / "contract" / "shm" / "MeshControlInterface.scala")
  ))
}

object emit extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def framework = library
  override def mainClass = Some("framework.tlp.shm.Emit")
}

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.tlp.shm.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / "SHM.scala"),
    PathRef(millSourcePath / "Banks.scala")
  ) ++ implementation()
)

def emitSources = T.sources(
  Seq(PathRef(millSourcePath / "src" / "main" / "scala" / "Emit.scala"))
)
