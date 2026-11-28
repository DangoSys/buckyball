package framework.device.peripherals.bootrom

import framework.device.peripherals.bootrom.configs.BootRomParams
import framework.generic.bus.chi.Params

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))
  val image                                             = Seq.tabulate(v("imageBytes").num.toInt)(i => (37 * i + 11).toByte)
  val p                                                 = BootRomParams(BigInt(v("base").num.toLong), BigInt(v("bytes").num.toLong), image)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BootRomLines(p, Params(addressBits = v("addressBits").num.toInt)),
    args = Array("--target-dir", s"$output/BootRomLines", "--split-verilog")
  )
}
