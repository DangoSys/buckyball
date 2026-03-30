package sims.bebop

import _root_.circt.stage.ChiselStage

/** `mill buckyball.runMain sims.bebop.EmitBebopSpikeCosimVerilog <abs-or-rel-dir>` */
object EmitBebopSpikeCosimVerilog {
  def main(args: Array[String]): Unit = {
    val dir = if (args.nonEmpty) args(0) else "gen-bebop-cosim"
    ChiselStage.emitSystemVerilogFile(
      new BebopSpikeCosimTop,
      firtoolOpts = Array.empty,
      args = Array("-td", dir),
    )
    ChiselStage.emitSystemVerilogFile(
      new VecComputeTop,
      firtoolOpts = Array.empty,
      args = Array("-td", dir),
    )
    println(s"EmitBebopSpikeCosimVerilog: wrote under $dir")
  }
}
