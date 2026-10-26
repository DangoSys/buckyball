package framework.core.tbus.router

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val frontend                                             = upickle.default.read[FrontendParam](values("frontend"))
  val memory                                               = upickle.default.read[MemDomainParam](values("memory"))
  val bus                                                  = upickle.default.read[framework.core.tbus.configs.Params](values("tbus"))
  val endpointIds                                          = values("endpointIds").arr.toSeq.map(_.num.toInt)
  val traceIds                                             = values("traceIds").arr.toSeq.map(_.num.toInt)

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define TR_BANKS ${memory.sharedBankNum}",
    s"`define TR_ROWS ${memory.sharedBankEntries}",
    s"`define TR_CLIENTS ${memory.sharedInputChannels}",
    s"`define TR_THREADS ${bus.slots}",
    s"`define TR_EXTERNAL ${bus.router.physicalPorts}",
    s"`define TR_DATA_BITS ${memory.bankWidth}",
    s"`define TR_MASK_BITS ${memory.bankMaskLen}",
    s"`define TR_TAG_BITS ${math.max(1, chisel3.util.log2Ceil(frontend.rob_entries))}",
    s"`define TR_ALIASES ${2 * memory.virtualBankCount + frontend.rob_entries}",
    s"`define TR_OWNERS ${endpointIds.mkString("'{", ",", "}")}"
  )

  val ports       = scala.collection.mutable.ArrayBuffer[String]()
  val leavesWidth = scala.collection.mutable.Map[String, Int]()
  val offsets     = scala.collection.mutable.Map[String, Int]()
  var inOffset    = 0; var outOffset = 0

  def leaves(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        leaves(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => leaves(field, s"${path}_$index") }
    case field => if (field.getWidth > 0) Seq(path -> field) else Seq.empty
  }

  Files.createDirectories(Paths.get(directory))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new Router(frontend, memory, bus, endpointIds, traceIds, values("diffTest").bool)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal        = if (input) "inputs" else "outputs";
        val offset    = if (input) inOffset else outOffset; val width = field.getWidth
        defs += s"`define TR_${direction}_${name.toUpperCase}_OFFSET $offset";
        defs += s"`define TR_${direction}_${name.toUpperCase}_WIDTH $width"; offsets(name) = offset;
        leavesWidth(name) = width
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        if (input) inOffset += width else outOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/Router", "--split-verilog")
  )
  defs += s"`define TR_INPUT_BITS $inOffset"; defs += s"`define TR_OUTPUT_BITS $outOffset"
  val clientNames = (0 until memory.sharedInputChannels).map(i => s"mem_req_$i")

  val clientFields = Seq(
    "read_req_valid",
    "read_req_ready",
    "read_req_bits_addr",
    "read_resp_valid",
    "read_resp_ready",
    "read_resp_bits_data",
    "write_req_valid",
    "write_req_ready",
    "write_req_bits_addr",
    "write_req_bits_data",
    "write_req_bits_mask_0",
    "write_resp_valid",
    "write_resp_ready",
    "write_resp_bits_ok",
    "bank_id",
    "group_id",
    "thread_id"
  )

  val meshNames = (0 until memory.sharedInputChannels).map(i =>
    s"memory_channels_$i"
  ) ++ (0 until bus.router.physicalPorts).map(i => s"memory_external_$i")

  val meshFields = Seq(
    "request_valid",
    "request_ready",
    "request_bits_tdest",
    "request_bits_tuser",
    "request_bits_tdata",
    "request_bits_tkeep",
    "request_bits_tid",
    "request_bits_tlast",
    "response_valid",
    "response_ready",
    "response_bits_tdata",
    "response_bits_tkeep",
    "response_bits_tid",
    "response_bits_tuser",
    "response_bits_tlast"
  )

  val mappingNames = (0 until bus.slots).map(i => s"mapping_$i")

  val mappingFields = Seq(
    "valid",
    "ready",
    "bits_bank",
    "bits_vbank",
    "bits_previous",
    "bits_threadId",
    "bits_allocate",
    "bits_reuse",
    "bits_retire",
    "bits_write"
  )

  def table(names: Seq[String], fields: Seq[String]) =
    names.map(n => fields.map(f => offsets(s"${n}_$f")).mkString("'{", ",", "}")).mkString("'{", ",", "}")
  defs += s"`define TR_CLIENT_OFFSETS ${table(clientNames, clientFields)}"
  defs += s"`define TR_MESH_OFFSETS ${table(meshNames, meshFields)}"
  defs += s"`define TR_MAPPING_OFFSETS ${table(mappingNames, mappingFields)}"
  val copies = scala.collection.mutable.ArrayBuffer[String]()
  for ((name, offset) <- offsets.toSeq.sortBy(_._1) if name.startsWith("memory_localBanks_")) {
    val other       = "localBanks_" + name.stripPrefix("memory_localBanks_")
    val memoryInput = name.contains("_request_") && !name.endsWith("_ready") || name.contains("_response_ready")
    val from        = if (memoryInput) name else other; val to = if (memoryInput) other else name
    val gate        =
      if (name.endsWith("_ready") || name.endsWith("_valid")) -1
      else offsets(from.take(from.indexOf("_bits")) + "_valid")
    val width       = leavesWidth(name)
    copies += s"'{${offsets(from)},${offsets(to)},$width,$gate}"
  }
  defs += s"`define TR_LOCAL_COPIES ${copies.mkString("'{", ",", "}")}"
  defs += s"`define TR_LOCAL_FIELDS ${copies.size}"
  Files.writeString(Paths.get(directory, "parameters.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "ports.svh"), ports.mkString(",\n") + "\n")
}
