import mill._
import $file.^.^.^.^.^.^.settings
import $file.^.^.^.^.^.^.thirdparty.{build => thirdparty_build}

val ipRoot: os.Path = millSourcePath
val ip = millSourcePath / os.up / os.up / "ip"
val lib = millSourcePath / os.up / os.up / "lib"

def implementation = T.sources(
  os.walk(millSourcePath / "src" / "main" / "scala").filter(path =>
    os.isFile(path) && path.ext == "scala" && path.last != "Emit.scala"
  ).map(PathRef(_))
)

object library extends settings.FrameworkModule {
  override def moduleRoot = ipRoot
  override def moduleDeps = Seq(thirdparty_build.protoJava)
  override def sources = T.sources(implementation() ++ Seq(
    PathRef(ip / "axis" / "src" / "main" / "scala" / "Interface.scala"),
    PathRef(ip / "bank" / "src" / "main" / "scala" / "Bank.scala"),
    PathRef(ip / "bank" / "src" / "main" / "scala" / "BankSet.scala"),
    PathRef(ip / "axi4" / "src" / "main" / "scala" / "Params.scala"),
    PathRef(ip / "axi4" / "src" / "main" / "scala" / "Interface.scala"),
    PathRef(ip / "axi4" / "src" / "main" / "scala" / "Interconnect.scala"),
    PathRef(ipRoot / os.up / os.up / "thread" / "tlink" / "Dma.scala"),
    PathRef(lib / "queue" / "Queue.scala"),
    PathRef(lib / "arbiter" / "Arbiter.scala")
  ))
}

object emit extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def framework = library
  override def mainClass = Some("framework.core.shm.Emit")
}

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.core.shm.Emit")
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
