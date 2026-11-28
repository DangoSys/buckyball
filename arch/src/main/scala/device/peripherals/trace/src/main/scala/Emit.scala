package framework.device.peripherals.trace

import framework.device.peripherals.trace.configs.TraceParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Trace(TraceParams(BigInt(v("base").num.toLong), BigInt(v("bytes").num.toLong), v("stride").num.toInt)),
    args = Array("--target-dir", s"$output/Trace", "--split-verilog")
  )
}
