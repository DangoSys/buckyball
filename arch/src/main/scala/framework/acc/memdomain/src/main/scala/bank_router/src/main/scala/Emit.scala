package framework.acc.memdomain.bank_router

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
  val gp                                                   = upickle.default.read[GpDomainParam](values("gp"))

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define BR_BANKS ${memory.bankNum}",
    s"`define BR_ROWS ${memory.bankEntries}",
    s"`define BR_CLIENTS ${memory.bankRequestPorts + 2}",
    s"`define BR_ALIASES ${2 * memory.virtualBankCount + frontend.rob_entries}",
    s"`define BR_DATA_BITS ${memory.bankWidth}",
    s"`define BR_MASK_BITS ${memory.bankMaskLen}"
  )

  val ports    = scala.collection.mutable.ArrayBuffer[String]()
  val offsets  = scala.collection.mutable.Map[String, Int]()
  var inOffset = 0; var outOffset = 0

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
      val dut = new BankRouter(frontend, memory, gp, values("diffTest").bool)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal        = if (input) "inputs" else "outputs";
        val offset    = if (input) inOffset else outOffset; val width = field.getWidth
        defs += s"`define BR_${direction}_${name.toUpperCase}_OFFSET $offset";
        defs += s"`define BR_${direction}_${name.toUpperCase}_WIDTH $width"; offsets(name) = offset
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        if (input) inOffset += width else outOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/BankRouter", "--split-verilog")
  )
  defs += s"`define BR_INPUT_BITS $inOffset"; defs += s"`define BR_OUTPUT_BITS $outOffset"
  val clientNames =
    (0 until memory.bankRequestPorts).map(i => s"requests_$i") ++ (0 until 2).map(i => s"directRequest_$i")

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

  val bankFields = Seq(
    "sramRead_req_valid",
    "sramRead_req_ready",
    "sramRead_req_bits_addr",
    "sramRead_resp_valid",
    "sramRead_resp_ready",
    "sramRead_resp_bits_data",
    "sramWrite_req_valid",
    "sramWrite_req_ready",
    "sramWrite_req_bits_addr",
    "sramWrite_req_bits_data",
    "sramWrite_req_bits_mask_0",
    "sramWrite_resp_valid",
    "sramWrite_resp_ready",
    "sramWrite_resp_bits_ok",
    "clear",
    "clearing",
    "bind"
  )

  def table(names: Seq[String], fields: Seq[String]) =
    names.map(n => fields.map(f => offsets(s"${n}_$f")).mkString("'{", ",", "}")).mkString("'{", ",", "}")
  defs += s"`define BR_CLIENT_OFFSETS ${table(clientNames, clientFields)}"
  defs += s"`define BR_BANK_OFFSETS ${table((0 until memory.bankNum).map(i => s"physical_$i"), bankFields)}"
  Files.writeString(Paths.get(directory, "parameters.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "ports.svh"), ports.mkString(",\n") + "\n")
}
