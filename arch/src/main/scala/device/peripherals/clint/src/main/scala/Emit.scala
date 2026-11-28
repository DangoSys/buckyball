package framework.device.peripherals.clint

import framework.device.peripherals.clint.configs.ClintParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))

  val p = ClintParams(
    v("hartIds").arr.toSeq.map(_.num.toInt),
    BigInt(v("base").num.toLong),
    BigInt(v("bytes").num.toLong),
    v("tickCycles").num.toInt
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Clint(p),
    args = Array("--target-dir", s"$output/Clint", "--split-verilog")
  )
}
