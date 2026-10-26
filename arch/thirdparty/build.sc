import mill._
import mill.scalalib._
val archRoot: os.Path = millSourcePath / os.up

object protoJava extends JavaModule {

  def protoDir = T {
    archRoot / os.up / "bbdev" / "api" / "steps" / "config" / "scripts" / "proto"
  }

  def protoSources = T.sources {
    Seq(PathRef(protoDir() / "design.proto"), PathRef(protoDir() / "hardware.proto")) ++
      Seq("balls", "cores").flatMap(kind =>
        os.walk(archRoot / os.up / "examples" / "components" / kind).filter(p => p.ext == "proto" && !p.toString.contains("/thirdparty/") && !p.toString.contains("/out/")).map(PathRef(_)))
  }

  def generatedSources = T {
    val dir   = protoDir()
    val protos = protoSources().map(_.path)
    val command = Seq("protoc", s"--java_out=${T.dest}") ++
      protos.map(path => s"-I${path / os.up}").distinct ++ protos.map(_.toString)
    os.proc(command).call()
    Seq(PathRef(T.dest))
  }

  override def ivyDeps = Agg(ivy"com.google.protobuf:protobuf-java:4.35.1")
}

object hardfloat extends SbtModule {
  override def millSourcePath =
    archRoot / "thirdparty" / "berkeley-hardfloat"
  override def scalaVersion   = "2.13.16"
  override def scalacOptions = Seq("-Ymacro-annotations")

  // Override sources to match build.sbt behavior
  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(millSourcePath / "hardfloat" / "src" / "main" / "scala")
    )
  }

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}

object midas_target_utils extends SbtModule {
  override def millSourcePath =
    archRoot / os.up / "thirdparty" / "firesim" / "sim" / "midas" / "targetutils"
  override def scalaVersion   = "2.13.16"

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}

object gemmini extends SbtModule {
  override def millSourcePath =
    archRoot / "thirdparty" / "gemmini"
  override def scalaVersion   = "2.13.16"
  override def scalacOptions = Seq("-Ymacro-annotations")

  override def sources = T.sources {
    Seq(
      "Arithmetic",
      "Dataflow",
      "MeshWithDelays",
      "Mesh",
      "Tile",
      "PE",
      "Util",
      "Transposer",
      "TagQueue",
      "Pipeline",
      "Shifter",
      "SyncMem"
    )
      .map(name => PathRef(millSourcePath / "src" / "main" / "scala" / "gemmini" / s"$name.scala"))
  }

  override def moduleDeps = Seq(hardfloat)

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}
