package framework.generic.gvn

import java.nio.file.{Files, Paths}
import framework.generic.bus.axi4

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))

  val p = Params(
    values("buckets").num.toInt,
    values("entries").num.toInt,
    axi4.Params(values("addressBits").num.toInt, values("dataBits").num.toInt, values("idBits").num.toInt)
  )

  Files.createDirectories(Paths.get(output))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CheckerOnChip(p),
    args = Array("--target-dir", output),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
