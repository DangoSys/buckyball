package framework.ip.bus.axi

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val parameters                                        = ujson.read(os.read(os.Path(config)))
  val dataBits                                          = parameters("dataBits").num.toInt
  val inputs                                            = parameters("inputs").num.toInt
  require(inputs == 2, "The current UVM arbiter interface requires two inputs")
  require(dataBits == 16 || dataBits == 32, "The current UVM reference supports 16 or 32 bits")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Master(dataBits = dataBits),
    args = Array("--target-dir", s"$output/Master", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Slave(dataBits = dataBits),
    args = Array("--target-dir", s"$output/Slave", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PacketArbiter(inputs = inputs, dataBits = dataBits),
    args = Array("--target-dir", s"$output/PacketArbiter", "--split-verilog")
  )
}
