package framework.ip.memory.spm

import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val p                                                 = Params(BigInt(values("base").str), values("bytes").num.toInt, values("dataBits").num.toInt)
  val directory                                         = Paths.get(output, "Memory")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Memory(p),
    args = Array("--target-dir", directory.toString, "--split-verilog")
  )
  Files.writeString(
    directory.resolve("parameters.svh"),
    s"""`ifndef SPM_PARAMETERS
`define SPM_PARAMETERS
`define SPM_BASE 64'h${p.base.toString(16)}
`define SPM_BYTES ${p.bytes}
`define SPM_BITS ${p.dataBits}
`endif
"""
  )
}
