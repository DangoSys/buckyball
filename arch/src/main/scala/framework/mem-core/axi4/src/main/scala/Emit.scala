package memcore.bus.axi4

object Emit extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Interconnect(Params(), masters = 2),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Interconnect", "--split-verilog")
  )
}
