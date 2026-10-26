import mill._
import mill.scalalib._
import $file.^.^.^.^.^.arch.thirdparty.{build => thirdparty}
import $file.^.^.^.^.^.arch.src.main.scala.framework.{build => framework}

val root: os.Path = millSourcePath / os.up
trait ScalaBuild extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}
object cde extends ScalaBuild {
  override def sources = T.sources(root / "thirdparty" / "rocket-chip" / "dependencies" / "cde" / "cde" / "src")
}
object pipeline extends ScalaBuild {
  override def moduleDeps = Seq(cde, thirdparty.hardfloat, thirdparty.midas_target_utils, framework.library)
  override def sources = T.sources {
    val src = root / "thirdparty" / "rocket-chip" / "src" / "main" / "scala"
    val files = Seq(
      "rocket/ALU.scala", "rocket/AMOALU.scala", "rocket/BTB.scala", "rocket/Breakpoint.scala",
      "rocket/CSR.scala", "rocket/Consts.scala", "rocket/CustomInstructions.scala", "rocket/DebugROB.scala",
      "rocket/Decode.scala", "rocket/Events.scala", "rocket/IBuf.scala", "rocket/IDecode.scala",
      "rocket/Instructions.scala", "rocket/Instructions32.scala", "rocket/Multiplier.scala", "rocket/PMP.scala",
      "rocket/RVC.scala", "rocket/package.scala", "tile/FPU.scala", "tile/CustomCSRs.scala",
      "util/Misc.scala", "util/ClockGate.scala", "util/CoreMonitor.scala", "util/Property.scala",
      "util/Counters.scala", "util/ECC.scala", "util/Replacement.scala", "util/ShiftReg.scala",
      "util/ShiftQueue.scala", "util/HellaQueue.scala", "util/LCG.scala", "util/PlusArg.scala",
      "util/MuxLiteral.scala", "util/Arbiters.scala", "util/AsyncResetReg.scala", "util/Timer.scala",
      "util/GeneratorUtils.scala", "unittest/UnitTest.scala"
    )
    files.map(f => PathRef(src / os.RelPath(f))) ++ Seq(PathRef(root / "arch" / "pipeline"))
  }
  override def resources = T.sources(root / "thirdparty" / "rocket-chip" / "src" / "main" / "resources")
}
object library extends ScalaBuild {
  override def moduleDeps = Seq(pipeline, framework.library)
  override def sources = T.sources {
    os.walk(root / "arch").filter(p => p.ext == "scala" && !p.toString.contains("/pipeline/")).map(PathRef(_)) ++
      Seq(PathRef(root / "configs"))
  }
}
