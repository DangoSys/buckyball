import mill._
import mill.scalalib._
import $file.^.^.^.^.^.arch.thirdparty.{build => thirdparty}
import $file.^.^.^.^.^.arch.src.main.scala.{build => framework}
val root = millSourcePath / os.up
val rc = root / os.up / "rocket" / "thirdparty" / "rocket-chip"
val gemminiRoot = root / "thirdparty" / "gemmini"
trait ScalaBuild extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}
object cde extends ScalaBuild { override def sources = T.sources(rc / os.RelPath("dependencies/cde/cde/src")) }
object macros extends ScalaBuild {
  override def sources = T.sources(rc / os.RelPath("macros/src/main/scala"))
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"org.scala-lang:scala-reflect:2.13.16")
}
object diplomacy extends ScalaBuild {
  override def sources = T.sources(rc / os.RelPath("dependencies/diplomacy/diplomacy/src"))
  override def moduleDeps = Seq(cde)
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"com.lihaoyi::sourcecode:0.3.1")
}
object hardfloat extends ScalaBuild { override def sources = T.sources(rc / os.RelPath("dependencies/hardfloat/hardfloat/src/main/scala")) }
object rocketchip extends ScalaBuild {
  override def sources = T.sources(rc / os.RelPath("src/main/scala"))
  override def resources = T.sources(rc / os.RelPath("src/main/resources"))
  override def moduleDeps = Seq(cde, macros, diplomacy, hardfloat)
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"com.lihaoyi::mainargs:0.5.0", ivy"org.json4s::json4s-jackson:4.0.5")
}
object native extends ScalaBuild {
  override def sources = T.sources(gemminiRoot / os.RelPath("src/main/scala"), root / "arch" / "native")
  // arch/native/ReservationStation.scala replaces the upstream one.
  override def allSourceFiles = T {
    super.allSourceFiles().filterNot(_.path == gemminiRoot / os.RelPath("src/main/scala/gemmini/ReservationStation.scala"))
  }
  override def resources = T.sources(gemminiRoot / os.RelPath("src/main/resources"))
  override def moduleDeps = Seq(rocketchip, thirdparty.midas_target_utils)
  override def forkArgs = Seq("-Xmx32G", "-Xss64m")
}
def nativeRtl = T {
  val cp = native.runClasspath().map(_.path)
  os.proc("java", "-Xmx32G", "-Xss64m", "-cp", cp.mkString(java.io.File.pathSeparator),
    "examples.hart.gemmini.native.Emit", "--target-dir", T.dest / "gemmini").call(cwd = T.dest)
  PathRef(T.dest)
}

// The parameter message of this hart.
object parameters extends JavaModule {

  def protoSource = T.source(root / "configs" / "gemminihart_params.proto")

  def generatedSources = T {
    val proto = protoSource().path
    os.proc("protoc", s"--java_out=${T.dest}", s"-I${proto / os.up}", proto).call()
    Seq(PathRef(T.dest))
  }

  override def ivyDeps = Agg(ivy"com.google.protobuf:protobuf-java:4.35.1")
}
object library extends ScalaBuild {
  override def moduleDeps = Seq(parameters, framework.library)
  override def sources = T.sources {
    os.walk(root / "arch").filter(p => p.ext == "scala" && !p.startsWith(root / "arch" / "native")).map(PathRef(_)) ++
      Seq(PathRef(root / "configs"))
  }
  override def resources = T.sources(Seq(nativeRtl()))
}
