import mill._
import mill.scalalib._
import $file.^.^.^.^.^.arch.src.main.scala.{build => framework}
import $file.^.^.rocket.arch.{build => rocket}
val root = millSourcePath / os.up
object parameters extends JavaModule {
  def protoSource = T.source(root / "configs" / "magia_params.proto")
  def generatedSources = T {
    val proto = protoSource().path
    os.proc("protoc", s"--java_out=${T.dest}", s"-I${proto / os.up}", proto).call()
    Seq(PathRef(T.dest))
  }
  override def ivyDeps = Agg(ivy"com.google.protobuf:protobuf-java:4.35.1")
}
def nativeInputs = T.sources(root / "arch" / "scripts" / "prepare.py", root / "arch" / "tile", root / "arch" / "MagiaMemory.sv", root / "arch" / "ClockGate.sv",
  root / "arch" / "MeshAXI.sv", root / "arch" / "Mesh.sv", root / "arch" / "NoC.sv", root / "arch" / "NoCPackage.sv", root / "arch" / "Tile.sv", root / "arch" / "hierarchy.vlt", root / "arch" / "verilator.f", root / "arch" / "hierarchy_params.v", root / "thirdparty" / "MAGIA" / "Bender.lock",
  root / "thirdparty" / "MAGIA" / "Bender.yml")
def nativeRtl = T {
  nativeInputs()
  os.proc("nix", "develop", s"${root / os.up / os.up / os.up / os.up}#magia", "-c",
    "python3", root / "arch" / "scripts" / "prepare.py", T.dest).call()
  PathRef(T.dest)
}
object library extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(parameters, framework.library, rocket.library)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources = T.sources(root / "arch", root / "configs")
  override def resources = T.sources(Seq(nativeRtl()))
}
