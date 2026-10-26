package framework.ip.memory.bank

import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))

  val p = BankSetParams(
    values("dataBits").num.toInt,
    values("banks").num.toInt,
    values("entriesPerBank").num.toInt,
    values("tagBits").num.toInt
  )

  require(p.dataBits == 32, "The Bank reference model uses 32-bit words")
  Files.createDirectories(Paths.get(output))
  Files.writeString(
    Paths.get(output, "parameters.svh"),
    s"""`ifndef BANK_PARAMETERS
`define BANK_PARAMETERS
`define BANK_ROWS ${p.entriesPerBank}
`define BANK_COUNT ${p.banks}
`define BANK_WORDS ${p.entriesPerBank * p.banks}
`define BANK_ROW_BITS ${p.rowBits}
`define BANK_ADDRESS_BITS ${p.addressBits}
`define BANK_TAG_BITS ${p.tagBits}
`endif
"""
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Bank(p),
    args = Array("--target-dir", s"$output/Bank", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BankSet(p),
    args = Array("--target-dir", s"$output/BankSet", "--split-verilog")
  )
}
