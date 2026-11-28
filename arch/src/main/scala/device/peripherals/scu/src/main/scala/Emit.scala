package framework.device.peripherals.scu

import framework.device.peripherals.scu.configs.SCUParams
import framework.generic.contract.memory.access.AccessParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val v                                                 = ujson.read(os.read(os.Path(config)))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new SystemControl(
      SCUParams(
        BigInt(v("base").num.toLong),
        BigInt(v("stride").num.toLong),
        BigInt(v("bytes").num.toLong),
        v("maxHarts").num.toInt
      ),
      AccessParams(tagBits = v("tagBits").num.toInt)
    ),
    args = Array("--target-dir", s"$output/SystemControl", "--split-verilog")
  )
}
