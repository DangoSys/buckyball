import mill._
import $file.^.^.^.^.^.^.settings


val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.ip.memory.mesh_shm.Emit")
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshBankNode.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshChannel.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshChannelOrdering.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshControlInterface.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshCreditLink.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshRouter.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshSharedMem.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshSharedMemInterface.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshSharedMemParams.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshThreadEndpoint.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/MeshTransferController.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Geometry.scala"))
  )
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  )
)
