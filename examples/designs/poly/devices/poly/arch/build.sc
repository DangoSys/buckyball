import mill._
import mill.scalalib._
import $file.^.^.^.^.^.^.arch.src.main.scala.hart.cbus.{build => cbus_build}
import $file.^.^.^.^.^.^.arch.src.main.scala.thread.tbus.{build => tbus_build}
import $file.^.^.^.^.^.{build => examples}

// The design: the arch directories of its device and harts.
val designRoot: os.Path = millSourcePath / os.up / os.up / os.up

object ballParameters extends JavaModule {

  def protoSources = T.sources(
    designRoot / "balls" / "mxint8unpack" / "configs" / "mxint8unpack_params.proto"
  )

  def generatedSources = T {
    val protos = protoSources().map(_.path)
    os.proc(
      Seq("protoc", s"--java_out=${T.dest}") ++ protos.map(path => s"-I${path / os.up}").distinct ++
        protos.map(_.toString)
    ).call()
    Seq(PathRef(T.dest))
  }

  override def ivyDeps = Agg(ivy"com.google.protobuf:protobuf-java:4.35.1")
}

object design extends SbtModule {

  override def generatedSources = T {
    cbus_build.registry(sources(), T.dest) ++
      tbus_build.registry(sources(), designRoot / "threads", T.dest, "examples.designs.poly.harts.poly.generated")
  }

  override def millSourcePath      = designRoot
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(examples.library, ballParameters)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")

  override def sources = T.sources(
    designRoot / "devices" / "poly" / "arch",
    designRoot / "harts" / "poly" / "arch",
    designRoot / "harts" / "reduce-unpack" / "arch",
    designRoot / "balls" / "mxint8unpack" / "arch",
    designRoot / "threads" / "ffn" / "arch",
    designRoot / "threads" / "softmax" / "arch",
    designRoot / "threads" / "mixed5" / "arch",
    designRoot / "threads" / "ffn5" / "arch"
  )

}
