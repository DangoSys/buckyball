import mill._
import $file.^.^.^.^.^.^.settings
import $file.^.^.^.^.^.^.thirdparty.{build => thirdparty_build}

val root: os.Path = millSourcePath

object library extends settings.FrameworkModule {
  override def moduleRoot = root
  override def moduleDeps = Seq(thirdparty_build.protoJava)
  override def sources = T.sources(sourceFiles() ++ Seq(
    PathRef(root / os.up / os.up / "ip" / "axi4" / "src" / "main" / "scala" / "Params.scala"),
    PathRef(root / os.up / os.up / "ip" / "axi4" / "src" / "main" / "scala" / "Interface.scala"),
    PathRef(root / os.up / os.up / "lib" / "queue" / "Queue.scala")
  ))
}

object emit extends settings.IpModule {
  override def moduleRoot = root
  override def framework = library
  override def mainClass = Some("framework.device.shm.Emit")
}

def sourceFiles = T.sources(
  Seq(
    PathRef(millSourcePath / "C2C.scala"),
    PathRef(millSourcePath / "Control.scala"),
    PathRef(millSourcePath / "SharedAxiEndpoint.scala"),
    PathRef(millSourcePath / "StorageAccess.scala"),
    PathRef(millSourcePath / "Storage.scala"),
    PathRef(millSourcePath / "Node.scala"),
    PathRef(millSourcePath / "Router.scala")
  )
)

def sources = sourceFiles

def emitSources = T.sources(Seq.empty[PathRef])
