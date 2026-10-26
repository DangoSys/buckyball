package framework.core.memory.access.cache

import java.nio.file.{Files, Paths}
import framework.ip.bus.chi.{Opcode, RequestFlit, ResponseFlit, SnoopFlit}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val p                                                    = upickle.default.read[configs.Params](Files.readString(Paths.get(config)))
  val c                                                    = p.cache.chi
  Files.createDirectories(Paths.get(directory))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Cache(p),
    args = Array("--target-dir", s"$directory/Cache", "--split-verilog")
  )
  Files.writeString(
    Paths.get(directory, "parameters.svh"),
    s"""`define CHI_NODE_BITS ${c.nodeIdBits}
`define CHI_ADDRESS_BITS ${c.addressBits}
`define CHI_DATA_BITS ${c.dataBits}
`define CHI_REQ_BITS ${(new RequestFlit(c)).flitWidth}
`define CHI_RSP_BITS ${(new ResponseFlit(c)).flitWidth}
`define CHI_SNP_BITS ${(new SnoopFlit(c)).flitWidth}
`define CHI_NODE_ID ${p.cache.nodeId}
`define CHI_HOME_ID ${p.cache.homeId}
`define CHI_BANKS ${p.cache.banks}
`define CHI_CACHE_LINES ${p.cache.cacheLines}
`define CMO_ID_BITS ${p.tracking.idBits}
`define CMO_CLEAN ${Opcode.CleanShared}
`define CMO_CLEAN_INVALID ${Opcode.CleanInvalid}
`define CMO_INVALID ${Opcode.MakeInvalid}
`define CMO_COMP ${Opcode.Comp}
"""
  )
}
