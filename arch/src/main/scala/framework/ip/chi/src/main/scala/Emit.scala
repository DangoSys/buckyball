package framework.ip.bus.chi

import java.nio.file.{Files, Paths}
import framework.ip.bus.chi.snf.SramEndpoint
import framework.ip.bus.chi.rnf.{BankedChiCache, RnfParams}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val p                                                 = Params(values("nodeIdBits").num.toInt, values("addressBits").num.toInt, values("dataBits").num.toInt)
  val flitBits                                          = values("linkFlitBits").num.toInt
  val credits                                           = values("credits").num.toInt
  val node                                              = values("nodeId").num.toInt
  val home                                              = values("homeId").num.toInt
  val records                                           = values("records").num.toInt
  val banks                                             = values("banks").num.toInt
  val cacheLines                                        = values("cacheLines").num.toInt
  Files.createDirectories(Paths.get(output))
  Files.writeString(
    Paths.get(output, "parameters.svh"),
    s"""`ifndef CHI_PARAMETERS
`define CHI_PARAMETERS
`define CHI_NODE_BITS ${p.nodeIdBits}
`define CHI_ADDRESS_BITS ${p.addressBits}
`define CHI_DATA_BITS ${p.dataBits}
`define CHI_SRAM_MODULE SramEndpoint${p.dataBits}
`define CHI_REQ_BITS ${(new RequestFlit(p)).flitWidth}
`define CHI_RSP_BITS ${(new ResponseFlit(p)).flitWidth}
`define CHI_SNP_BITS ${(new SnoopFlit(p)).flitWidth}
`define CHI_LINK_BITS $flitBits
`define CHI_CREDITS $credits
`define CHI_CREDIT_BITS ${chisel3.util.log2Ceil(credits + 1)}
`define CHI_NODE_ID $node
`define CHI_HOME_ID $home
`define CHI_RECORDS $records
`define CHI_PENDING_BITS ${chisel3.util.log2Ceil(records + 1)}
`define CHI_BANKS $banks
`define CHI_CACHE_LINES $cacheLines
`define CHI_SRAM_SLOTS ${values("sramSlots").num.toInt}
`define CHI_SRAM_LINES ${values("sramLines").num.toInt}
`endif
"""
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Tx(flitBits, credits),
    args = Array("--target-dir", s"$output/Tx", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Rx(flitBits, credits),
    args = Array("--target-dir", s"$output/Rx", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new SramEndpoint(
      p,
      node,
      values("sramSlots").num.toInt,
      values("sramLines").num.toInt,
      values("sramRxDepth").num.toInt
    ),
    args = Array("--target-dir", s"$output/SramEndpoint", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new RequestRetry(p, node, records),
    args = Array("--target-dir", s"$output/RequestRetry", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BankedChiCache(RnfParams(
      p,
      node,
      cacheLines,
      home,
      values("homeCount").num.toInt,
      banks,
      values("lineResult").bool,
      values("probe").bool
    )),
    args = Array("--target-dir", s"$output/BankedChiCache", "--split-verilog")
  )
}
