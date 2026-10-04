package memcore.bus.axi

object Emit extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Master(dataBits = 32),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Master", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Slave(dataBits = 32),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Slave", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PacketArbiter(inputs = 2, dataBits = 32),
    firtoolOpts = args,
    args = Array("--target-dir", "build/PacketArbiter", "--split-verilog")
  )
}
