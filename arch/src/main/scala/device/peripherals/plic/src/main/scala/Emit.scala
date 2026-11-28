package framework.device.peripherals.plic

import framework.device.peripherals.plic.configs.PlicParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))

  val p = PlicParams(
    hartIds = v("hartIds").arr.toSeq.map(_.num.toInt),
    base = BigInt(v("base").num.toLong),
    bytes = BigInt(v("bytes").num.toLong),
    sources = v("sources").num.toInt,
    priorityBits = v("priorityBits").num.toInt
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Plic(p),
    args = Array("--target-dir", s"$output/Plic", "--split-verilog")
  )
}
