import mill._
import mill.scalalib._
import $file.^.^.^.^.^.arch.src.main.scala.framework.{build => framework}

val componentRoot: os.Path = millSourcePath / os.up
val vendorRoot: os.Path    = componentRoot / "thirdparty" / "xiangshan"
// Our XiangShan sources; files under native/xiangshan replace the upstream file at the same path.
val nativeRoot: os.Path    = componentRoot / "arch" / "native"
val vendorHead: os.Path    =
  os.Path(os.proc("git", "rev-parse", "--git-path", "HEAD").call(cwd = vendorRoot).out.text().trim, vendorRoot)

def upstreamInputs = T.sources {
  val paths = Seq(
    "rocket-chip/src", "rocket-chip/macros/src", "rocket-chip/cde/cde/src", "rocket-chip/hardfloat/hardfloat/src",
    "difftest/src", "utility/src", "huancun/src", "fudian/src", "yunsuan/src", "coupledL2/src", "openLLC/src",
    "openLLC/openNCB/src", "ChiselAIA/src", "ChiselIOPMP/src", "macros/src", "build.sc", ".mill-version",
    "rocket-chip/common.sc", "rocket-chip/cde/common.sc", "rocket-chip/hardfloat/common.sc", "huancun/common.sc",
    "coupledL2/common.sc", "openLLC/common.sc"
  )
  paths.map(path => PathRef(vendorRoot / os.RelPath(path))) :+ PathRef(vendorHead)
}

// Libraries XiangShan depends on (rocket-chip, utility, coupledL2, ...) with their resources, built by XiangShan's own build.
def upstreamClasspath = T {
  upstreamInputs()
  Seq("xiangshan.transitiveLocalClasspath", "xiangshan.compileClasspath").flatMap { task =>
    val shown = os.proc("mill", "--no-server", "show", task).call(cwd = vendorRoot).out.text()
    ujson.read(shown).arr.map(entry => os.Path(entry.str.split(":").last))
  }.distinct.map(PathRef(_))
}

def gitState = T.input {
  val sha   = os.proc("git", "rev-parse", "HEAD").call(cwd = vendorRoot).out.trim()
  val dirty = os.proc("git", "status", "-uno", "--porcelain").call(cwd = vendorRoot).out.text().nonEmpty
  (sha, dirty)
}

// Resources XiangShan's own build generates: version string and the commit id embedded in the CSR.
def versionResources = T {
  val (sha, dirty) = gitState()
  os.write(T.dest / "publishVersion", sha.take(12) + "\n")
  os.write(T.dest / "gitStatus", s"SHA=$sha\ndirty=${if (dirty) 1 else 0}\n")
  PathRef(T.dest)
}

def vendorResources = T.sources(vendorRoot / "src" / "main" / "resources")

object native extends ScalaModule {
  override def scalaVersion        = "2.13.15"
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations", "-Ytasty-reader")
  override def unmanagedClasspath  = T(Agg.from(upstreamClasspath()))

  override def sources = T.sources {
    val upstream = vendorRoot / "src" / "main" / "scala"
    os.walk(upstream)
      .filter(p => p.ext == "scala" && !os.exists(nativeRoot / p.relativeTo(upstream)))
      .map(PathRef(_)) :+ PathRef(nativeRoot)
  }

  override def resources = T(versionResources() +: vendorResources())
}

def nativeRtl = T {
  val output = T.dest / "xiangshan"
  os.proc(
    "java", "-Xmx40G", "-Xss256m", "-cp", native.runClasspath().map(_.path).mkString(java.io.File.pathSeparator),
    "examples.core.xiangshan.native.Emit", "--target-dir", output, "--target", "systemverilog"
  ).call(cwd = T.dest, env = Map("PATH" -> s"${sys.env("PATH")}:${vendorRoot / "src" / "main" / "resources"}"))
  os.move(output / "extern_modules.sv", output / "XiangShanExtern.sv")
  os.write.over(output / "filelist.f", os.read(output / "filelist.f").replace("extern_modules.sv", "XiangShanExtern.sv"))
  PathRef(T.dest)
}

trait ScalaBuild extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}
object library extends ScalaBuild {
  override def moduleDeps = Seq(framework.library)
  override def sources = T.sources {
    os.walk(componentRoot / "arch").filter(p => p.ext == "scala" && !p.startsWith(nativeRoot)).map(PathRef(_)) ++
      Seq(PathRef(componentRoot / "configs"))
  }
  override def resources = T.sources(Seq(nativeRtl()))
}
