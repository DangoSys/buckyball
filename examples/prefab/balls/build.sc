import mill._
import mill.scalalib._

val ballRoot: os.Path = millSourcePath

def sources = T.sources {
  os.list(millSourcePath).filter(os.isDir).flatMap { root =>
    val source  = root / "arch"
    val configs = root / "configs"
    (if (os.exists(source))
       os.walk(source).filter(p => p.ext == "scala" && !p.segments.exists(Set("out", "generated", "target", "build")))
     else Seq.empty) ++
      (if (os.exists(configs)) os.walk(configs).filter(_.ext == "scala") else Seq.empty)
  }.map(PathRef(_))
}

// The parameter messages of the Balls in this directory.
object parameters extends JavaModule {

  def protoSources = T.sources(
    os.list(ballRoot).filter(os.isDir).flatMap { root =>
      val configs = root / "configs"
      if (os.exists(configs)) os.list(configs).filter(_.ext == "proto") else Seq.empty
    }.map(PathRef(_))
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
