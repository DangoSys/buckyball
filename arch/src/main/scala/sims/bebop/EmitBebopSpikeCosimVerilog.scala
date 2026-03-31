package sims.bebop

import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule.LazyModule

/** `mill buckyball.runMain sims.bebop.EmitBebopSpikeCosimVerilog <abs-or-rel-dir>` */
object EmitBebopSpikeCosimVerilog {
  def main(args: Array[String]): Unit = {
    val dir = if (args.nonEmpty) args(0) else "gen-bebop-cosim"
    implicit val p: Parameters = org.chipsalliance.cde.config.Parameters.empty
    val bbCosim = LazyModule(new BebopBuckyballSubsystemCosim()(p))
    ChiselStage.emitSystemVerilogFile(
      bbCosim.module,
      firtoolOpts = Array.empty,
      args = Array("-td", dir),
    )
    val bbPath = java.nio.file.Paths.get(dir, "BebopBuckyballSubsystemCosim.sv")
    val bbText = java.nio.file.Files.readString(bbPath)
    val marker = "// ----- 8< ----- FILE \"firrtl_black_box_resource_files.f\""
    val cut    = bbText.indexOf(marker)
    if (cut >= 0) {
      java.nio.file.Files.writeString(bbPath, bbText.substring(0, cut))
    }
    ChiselStage.emitSystemVerilogFile(
      new VecComputeTop,
      firtoolOpts = Array.empty,
      args = Array("-td", dir),
    )
    println(s"EmitBebopSpikeCosimVerilog: wrote under $dir")
  }
}
