package memcore.memory.coherence

import java.nio.file.{Files, Paths}
import memcore.bus.chi._
import memcore.memory.coherence.configs.CoherenceParams

object Emit extends App {
  val p           = CoherenceParams.load("configs/default.toml")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Coherence(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
  val definitions = scala.collection.mutable.ArrayBuffer[String]()
  val ports       = scala.collection.mutable.ArrayBuffer[String]()

  def fields(flit: Flit): Seq[(String, Int)] = flit.fieldsLSB.map { field =>
    flit.elements.find(_._2 eq field).get._1 -> field.getWidth
  }

  def channel(
    kind:     String,
    prefix:   String,
    instance: String,
    fs:       Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((name, width) <- fs if width > 0) {
      definitions += s"`define COH_${kind}_${name.toUpperCase}_OFFSET $offset"
      definitions += s"`define COH_${kind}_${name.toUpperCase}_WIDTH $width"
      ports += s".$prefix$name($instance.bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define COH_${kind}_WIDTH $offset"
  }

  channel("REQ", "io_req_bits_", "source_if", fields(new RequestFlit(p.chi)))
  channel("RSP", "io_rxRsp_bits_", "rx_rsp", fields(new ResponseFlit(p.chi)))
  channel("DAT", "io_rxDat_bits_", "rx_dat", fields(new DataFlit(p.chi)))
  channel("TXRSP", "io_rsp_bits_", "tx_rsp", fields(new ResponseFlit(p.chi)))
  channel("TXDAT", "io_dat_bits_", "sink_if", fields(new DataFlit(p.chi)))
  channel("SNP", "io_snp_bits_flit_", "tx_snp", fields(new SnoopFlit(p.chi)))
  val snpWidth = (new SnoopFlit(p.chi)).flitWidth
  ports += s".io_snp_bits_targetNode(tx_snp.bits[$snpWidth +: ${p.chi.nodeIdBits}])"
  definitions += s"`define COH_SNP_TARGET_OFFSET $snpWidth"
  definitions += s"`define COH_SNP_TARGET_WIDTH ${p.chi.nodeIdBits}"
  definitions += s"`define COH_SNP_CHANNEL_WIDTH ${snpWidth + p.chi.nodeIdBits}"
  channel(
    "MEMREQ",
    "io_memoryReq_bits_",
    "mem_req",
    Seq("id" -> p.chi.txnIdBits, "addr" -> p.chi.addressBits, "write" -> 1, "data" -> 512, "mask" -> 64)
  )
  channel("MEMRESP", "io_memoryResp_bits_", "mem_resp", Seq("id" -> p.chi.txnIdBits, "data" -> 512, "error" -> 1))
  for (
    (port, instance) <- Seq(
                          "req"        -> "source_if",
                          "rxRsp"      -> "rx_rsp",
                          "rxDat"      -> "rx_dat",
                          "rsp"        -> "tx_rsp",
                          "dat"        -> "sink_if",
                          "snp"        -> "tx_snp",
                          "memoryReq"  -> "mem_req",
                          "memoryResp" -> "mem_resp"
                        )
  ) {
    ports += s".io_${port}_valid($instance.valid)"
    ports += s".io_${port}_ready($instance.ready)"
  }
  for (
    (name, value) <- Seq(
                       "AGENTS"           -> p.agents,
                       "MSHRS"            -> p.mshrEntries,
                       "HOME"             -> p.homeId,
                       "SETS"             -> p.cache.sets,
                       "WAYS"             -> p.cache.ways,
                       "BEATS"            -> p.chi.beatsPerLine,
                       "DATA_BITS"        -> p.chi.dataBits,
                       "OUTSTANDING_BITS" -> chisel3.util.log2Ceil(p.mshrEntries + 1),
                       "READ_SHARED"      -> Opcode.ReadShared,
                       "READ_NSD"         -> Opcode.ReadNotSharedDirty,
                       "READ_UNIQUE"      -> Opcode.ReadUnique,
                       "EVICT"            -> Opcode.Evict,
                       "WRITEBACK"        -> Opcode.WriteBackFull,
                       "CLEAN_INVALID"    -> Opcode.CleanInvalid,
                       "COMP"             -> Opcode.Comp,
                       "COMP_DBID"        -> Opcode.CompDBIDResp,
                       "COMP_ACK"         -> Opcode.CompAck,
                       "COMP_DATA"        -> Opcode.CompData,
                       "SNP_RESP"         -> Opcode.SnpResp,
                       "SNP_DATA"         -> Opcode.SnpRespData,
                       "COPYBACK_DATA"    -> Opcode.CopyBackWriteData,
                       "SNP_UNIQUE"       -> Opcode.SnpUnique,
                       "SNP_INVALID"      -> Opcode.SnpCleanInvalid,
                       "SNP_SHARED"       -> Opcode.SnpNotSharedDirty
                     )
  ) {
    definitions += s"`define COH_$name $value"
  }
  Files.writeString(
    Paths.get("build/coherence_config.svh"),
    "`ifndef COHERENCE_CONFIG_SVH\n`define COHERENCE_CONFIG_SVH\n" + definitions.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(Paths.get("build/coherence_ports.svh"), ports.mkString(",\n") + "\n")
}
