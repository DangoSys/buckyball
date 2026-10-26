package framework.acc.memdomain.prm

import java.nio.file.{Files, Paths}
import framework.acc.memdomain.configs.MemDomainParam

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val p                                                 = upickle.default.read[MemDomainParam](values("memory"))
  val diffTest                                          = values("diffTest").bool
  Files.createDirectories(Paths.get(output))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PRM(p, diffTest),
    args = Array("--target-dir", s"$output/PRM", "--split-verilog")
  )
  Files.writeString(
    Paths.get(output, "parameters.svh"),
    s"""`ifndef PRM_PARAMETERS
`define PRM_PARAMETERS
`define PRM_BANKS ${p.bankNum}
`define PRM_ROWS ${p.bankEntries}
`define PRM_WIDTH ${p.bankWidth}
`define PRM_MASKS ${p.bankMaskLen}
`define PRM_DIFF_TEST ${if (diffTest) 1 else 0}
`endif
"""
  )

  val ports = (0 until p.bankNum).flatMap { bank =>
    val prefix = s"io_banks_$bank"
    val fields = Seq(
      "sramRead_req_valid"      -> "read_valid",
      "sramRead_req_ready"      -> "read_ready",
      "sramRead_req_bits_addr"  -> "read_addr",
      "sramRead_resp_valid"     -> "read_result_valid",
      "sramRead_resp_ready"     -> "read_result_ready",
      "sramRead_resp_bits_data" -> "read_data",
      "sramWrite_req_valid"     -> "write_valid",
      "sramWrite_req_ready"     -> "write_ready",
      "sramWrite_req_bits_addr" -> "write_addr",
      "sramWrite_req_bits_data" -> "write_data",
      "sramWrite_resp_valid"    -> "write_result_valid",
      "sramWrite_resp_ready"    -> "write_result_ready",
      "sramWrite_resp_bits_ok"  -> "write_ok",
      "clear"                   -> "clear",
      "clearing"                -> "clearing",
      "bind"                    -> "bind_generation"
    ).map { case (port, signal) => s".${prefix}_$port(bus.ports[$bank].$signal)" }
    val masks  =
      (0 until p.bankMaskLen).map(mask => s".${prefix}_sramWrite_req_bits_mask_$mask(bus.ports[$bank].write_mask[$mask])")
    fields ++ masks ++ (if (diffTest) Seq(s".io_hashes_$bank(bus.ports[$bank].hash)") else Seq.empty)
  }

  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n"))
}
