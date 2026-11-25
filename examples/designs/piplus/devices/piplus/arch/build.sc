import mill._
import mill.scalalib._
import $file.^.^.^.^.^.^.arch.src.main.scala.{build => framework_build}
import $file.^.^.^.^.^.^.arch.src.main.scala.hart.cbus.{build => cbus_build}
import $file.^.^.^.^.^.^.arch.src.main.scala.thread.tbus.{build => tbus_build}
import $file.^.^.^.^.^.^.arch.src.main.scala.ball.src.main.scala.bbus.{build => bbus_build}

// The design: the arch directories of its device, harts, threads and Balls.
val designRoot: os.Path = millSourcePath / os.up / os.up / os.up
val threadRoot: os.Path = designRoot / "threads"
val ballRoot:   os.Path = designRoot / "balls"

// The parameter messages of the design's Balls.
object ballParameters extends JavaModule {

  def protoSources = T.sources(
    ballRoot / "mxint8mm" / "configs" / "mxint8mm_params.proto",
    ballRoot / "mxint8quant" / "configs" / "mxint8quant_params.proto",
    ballRoot / "softmax" / "configs" / "softmax_params.proto",
    ballRoot / "layernorm" / "configs" / "layernorm_params.proto"
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
  override def generatedSources    = T {
    val threads = sources().filter(_.path.startsWith(threadRoot)).flatMap(source => os.walk(source.path))
    val balls   = sources().filter(_.path.startsWith(ballRoot)).flatMap(source => os.walk(source.path))
    cbus_build.registry(sources(), T.dest) ++
      tbus_build.registry(
        threads.filter(_.ext == "scala").map(PathRef(_)),
        threadRoot,
        T.dest,
        "examples.designs.piplus.harts.generated"
      ) ++
      bbus_build.registry(balls.filter(_.ext == "scala").map(PathRef(_)), ballRoot, T.dest)
  }
  override def millSourcePath      = designRoot
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(framework_build.library, ballParameters)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(
    designRoot / "devices" / "piplus" / "arch",
    designRoot / "harts" / "control" / "arch",
    designRoot / "harts" / "solo" / "arch",
    designRoot / "harts" / "dense" / "arch",
    designRoot / "harts" / "action" / "arch",
    threadRoot / "dense" / "arch",
    threadRoot / "action" / "arch",
    ballRoot / "mxint8mm" / "arch",
    ballRoot / "mxint8quant" / "arch",
    ballRoot / "softmax" / "arch",
    ballRoot / "layernorm" / "arch"
  )
}
