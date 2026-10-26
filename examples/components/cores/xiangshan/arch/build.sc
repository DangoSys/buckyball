import mill._
import mill.scalalib._
import $file.^.^.^.^.^.arch.src.main.scala.framework.{build => framework}

val componentRoot: os.Path = millSourcePath / os.up
val vendorRoot: os.Path = componentRoot / "thirdparty" / "xiangshan"
val vendorHead: os.Path = os.Path(os.proc("git", "rev-parse", "--git-path", "HEAD").call(cwd = vendorRoot).out.text().trim, vendorRoot)

def nativeInputs = T.sources {
  val paths = Seq(
    "src", "rocket-chip/src", "rocket-chip/macros/src", "rocket-chip/cde/cde/src",
    "rocket-chip/hardfloat/hardfloat/src", "difftest/src", "utility/src", "huancun/src",
    "fudian/src", "yunsuan/src", "coupledL2/src", "openLLC/src", "openLLC/openNCB/src",
    "ChiselAIA/src", "ChiselIOPMP/src", "build.sc", ".mill-version",
    "rocket-chip/common.sc", "rocket-chip/cde/common.sc", "rocket-chip/hardfloat/common.sc",
    "huancun/common.sc", "coupledL2/common.sc", "openLLC/common.sc"
  )
  paths.map(path => PathRef(vendorRoot / os.RelPath(path))) :+ PathRef(vendorHead)
}

def nativeRtl = T {
  nativeInputs()
  val output = T.dest / "xiangshan"
  os.proc("mill", "--no-server", "native.runMain", "buckyball.Emit",
    "--target-dir", output, "--target", "systemverilog").call(cwd = vendorRoot)
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
  override def sources = T.sources(componentRoot / "arch", componentRoot / "configs")
  override def resources = T.sources(Seq(nativeRtl()))
}
