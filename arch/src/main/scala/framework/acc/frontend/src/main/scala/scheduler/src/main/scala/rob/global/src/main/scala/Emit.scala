package framework.acc.frontend.scheduler.rob.global

import chisel3._
import java.nio.file.{Files, Paths}
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.frontend.scheduler.rob.global.configs.RobParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val frontend                                          = upickle.default.read[FrontendParam](values("frontend"))
  val memory                                            = upickle.default.read[MemDomainParam](values("memory"))
  val gp                                                = upickle.default.read[GpDomainParam](values("gp"))
  val permission                                        = upickle.default.read[PmpParams](values("permission"))
  val storage                                           = upickle.default.read[RobParams](values("rob"))
  val diffTest                                          = values("diffTest").bool
  Files.createDirectories(Paths.get(output))

  def fields(data: Data, path: String): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq(path -> leaf.getWidth)
  }

  for (name <- Seq("Rob", "GlobalROB")) {
    val prefix       = if (name == "Rob") "ROB" else "GROB"
    val ports        = scala.collection.mutable.ArrayBuffer[String]()
    val definitions  = scala.collection.mutable.ArrayBuffer[String]()
    val offsets      = scala.collection.mutable.Map[String, Int]()
    val widths       = scala.collection.mutable.Map[String, Int]()
    var inputOffset  = 0
    var outputOffset = 0
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      {
        val (module, io) =
          if (name == "Rob") {
            val dut = new Rob(storage); (dut, dut.io)
          } else {
            val dut = new GlobalROB(frontend, memory, gp, permission, diffTest); (dut, dut.io)
          }
        for {
          (root, data) <- io.elements.toSeq
          (field, width) <- fields(data, root)
        } {
          val input     =
            if (name == "Rob") Set("alloc", "issue", "complete", "retireReady").contains(root) && field != "alloc_ready"
            else field == "alloc_valid" || field.startsWith(
              "alloc_bits_"
            ) || field == "issue_ready" || field == "mapping_ready" ||
            field == "complete_valid" || field == "complete_bits" || field == "bat_ready" || field.startsWith(
              "bat_renamed_"
            ) ||
            field.startsWith("bat_stale_") || field.startsWith("scoreboard_hazard_")
          val direction = if (input) "IN" else "OUT"
          val signal    = if (input) "inputs" else "outputs"
          val offset    = if (input) inputOffset else outputOffset
          definitions += s"`define ${prefix}_${direction}_${field.toUpperCase}_OFFSET $offset"
          definitions += s"`define ${prefix}_${direction}_${field.toUpperCase}_WIDTH $width"
          ports += s".io_$field(bus.$signal[$offset +: $width])"
          offsets(field) = offset; widths(field) = width
          if (input) inputOffset += width else outputOffset += width
        }
        module
      },
      args = Array("--target-dir", s"$output/$name", "--split-verilog")
    )
    definitions += s"`define ${prefix}_INPUT_BITS $inputOffset"
    definitions += s"`define ${prefix}_OUTPUT_BITS $outputOffset"
    if (name == "Rob") {
      definitions += s"`define ROB_ENTRIES ${storage.entries}"
      definitions += s"`define ROB_DATA_BITS ${storage.dataBits}"
      definitions += s"`define ROB_RETIRE_WIDTH ${storage.retireWidth}"
      val state = (0 until storage.entries).map(i =>
        Seq("data", "valid", "issued", "completed", "dependencies").map(n => offsets(s"${n}_$i")).mkString(
          "'{",
          ",",
          "}"
        )
      ).mkString("'{", ",", "}")
      val ready = (0 until storage.entries).map(i => offsets(s"retireReady_$i")).mkString("'{", ",", "}")
      definitions += s"`define ROB_STATE_OFFSETS $state"
      definitions += s"`define ROB_READY_OFFSETS $ready"
    } else {
      definitions += s"`define GROB_ENTRIES ${frontend.rob_entries}"
      definitions += s"`define GROB_VBANKS ${memory.virtualBankCount}"
      definitions += s"`define GROB_BANK_BITS ${frontend.bank_id_len}"
      definitions += s"`define GROB_OWNER ${frontend.ownerId}"
      definitions += s"`define GROB_ALIASES ${2 * (memory.virtualBankCount + (if (gp.enable) 2 else 0)) + frontend.rob_entries}"
      definitions += s"`define GROB_RENAMING ${if (frontend.bank_alias_enable) 1 else 0}"
      definitions += s"`define GROB_LAYOUT ${if (memory.bankLayoutEnable) 1 else 0}"
      definitions += s"`define GROB_GP ${if (gp.enable) 1 else 0}"
      definitions += s"`define GROB_DIFF ${if (diffTest) 1 else 0}"
      for (
        (key, fieldPrefix) <- Seq(
                                "IN_OPERATION" -> "alloc_bits_operation_",
                                "OUT_COMMAND"  -> "issue_bits_cmd_",
                                "IN_CONTEXT"   -> "alloc_bits_context_",
                                "OUT_CONTEXT"  -> "issue_bits_context_",
                                "IN_RENAMED"   -> "bat_renamed_",
                                "OUT_RENAMED"  -> "issue_bits_renamedBankAccess_"
                              )
      ) {
        val fs = offsets.keys.filter(_.startsWith(fieldPrefix)).toSeq
        definitions += s"`define GROB_${key}_OFFSET ${fs.map(offsets).min}"
        definitions += s"`define GROB_${key}_BITS ${fs.map(widths).sum}"
      }
      val issueFields = offsets.keys.filter(_.startsWith("issue_bits_")).toSeq
      definitions += s"`define GROB_ISSUE_OFFSET ${issueFields.map(offsets).min}"
      definitions += s"`define GROB_ISSUE_BITS ${issueFields.map(widths).sum}"
      val queryFields = Seq(
        "rd_bank_0_valid",
        "rd_bank_0_id",
        "rd_bank_1_valid",
        "rd_bank_1_id",
        "rd_wr_valid",
        "rd_wr_id",
        "wr_bank_valid",
        "wr_bank_id",
        "wr_update"
      )
      val queries     = (0 until frontend.rob_entries).map(i =>
        queryFields.map(n => offsets(s"scoreboard_query_${i}_$n")).mkString("'{", ",", "}")
      ).mkString("'{", ",", "}")
      definitions += s"`define GROB_QUERY_OFFSETS $queries"
      val baBase      = offsets.keys.filter(_.startsWith("bat_renamed_")).map(offsets).min
      val baOffsets   = queryFields.map(n => offsets(s"bat_renamed_$n") - baBase).mkString("'{", ",", "}")
      definitions += s"`define GROB_BA_OFFSETS $baOffsets"
      val traceFields =
        Seq("is_issue", "thread_id", "rob_id", "inst_id", "domain_id", "funct", "pc", "rs1_data", "rs2_data", "enable")
      val traces      = (0 until 4).map(i =>
        traceFields.map(n => offsets(s"traceInstructions_${i}_$n")).mkString("'{", ",", "}")
      ).mkString("'{", ",", "}")
      definitions += s"`define GROB_TRACE_OFFSETS $traces"

      for ((key, fieldPrefix) <- Seq("IN_ALLOC" -> "alloc_bits_", "MAPPING" -> "mapping_bits_")) {
        val fs = offsets.keys.filter(_.startsWith(fieldPrefix)).toSeq
        definitions += s"`define GROB_${key}_OFFSET ${fs.map(offsets).min}"
        definitions += s"`define GROB_${key}_BITS ${fs.map(widths).sum}"
      }

      val hazards = (0 until frontend.rob_entries).map(i => offsets(s"scoreboard_hazard_$i")).mkString("'{", ",", "}")
      definitions += s"`define GROB_HAZARD_OFFSETS $hazards"
    }
    Files.writeString(Paths.get(output, name, "parameters.svh"), definitions.mkString("\n") + "\n")
    Files.writeString(Paths.get(output, name, "ports.svh"), ports.mkString(",\n") + "\n")
  }
}
