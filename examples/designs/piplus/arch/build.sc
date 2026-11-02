import mill._
import mill.scalalib._
import $file.^.^.^.^.arch.src.main.scala.framework.{build => framework_build}
import $file.^.^.^.^.arch.src.main.scala.framework.device.cbus.{build => cbus_build}
import $file.^.^.^.^.arch.src.main.scala.framework.core.tbus.{build => tbus_build}
import $file.^.^.^.^.arch.src.main.scala.framework.acc.balldomain.src.main.scala.bbus.{build => bbus_build}

val designRoot: os.Path = millSourcePath / os.up

// The design's own threads and Balls: one component directory each.
val threadRoot: os.Path = designRoot / "threads"
val ballRoot:   os.Path = designRoot / "balls"

// The parameter messages of the design's Balls.
object ballParameters extends JavaModule {

  def protoSources = T.sources(
    os.list(ballRoot).flatMap(ball => os.list(ball / "configs").filter(_.ext == "proto")).map(PathRef(_))
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
      tbus_build.registry(threads.filter(_.ext == "scala").map(PathRef(_)), threadRoot, T.dest) ++
      bbus_build.registry(balls.filter(_.ext == "scala").map(PathRef(_)), ballRoot, T.dest)
  }
  override def millSourcePath      = designRoot
  override def forkWorkingDir      = T(designRoot)
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(framework_build.library, ballParameters)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(
    (designRoot / "arch" +: (os.list(threadRoot) ++ os.list(ballRoot)).map(_ / "arch").filter(os.isDir(_)))
      .map(PathRef(_))
  )
}
