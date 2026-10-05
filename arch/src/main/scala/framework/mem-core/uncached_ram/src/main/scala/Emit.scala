package memcore.memory.uncached_ram

import chisel3._
import java.nio.file.{Files, Paths}
import memcore.bus.chi.snf.{LineRequest, LineResponse}

object Emit extends App {
  val p     = Params()
  Files.createDirectories(Paths.get("build"))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Ram(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Ram", "--split-verilog")
  )
  val defs  = scala.collection.mutable.ArrayBuffer[String]()
  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:   String,
    name:   String,
    signal: String,
    data:   Bundle,
    define: Boolean = true
  ): Unit = {
    var offset = 0
    for ((field, value) <- data.elements.toSeq) {
      val width = value.getWidth
      if (define) {
        defs += s"`define RAM_${kind}_${field.toUpperCase}_OFFSET $offset";
        defs += s"`define RAM_${kind}_${field.toUpperCase}_WIDTH $width"
      }
      ports += s".io_${name}_bits_$field($signal.bits[$offset +: $width])"; offset += width
    }
    if (define) defs += s"`define RAM_${kind}_WIDTH $offset"
    ports += s".io_${name}_valid($signal.valid)"; ports += s".io_${name}_ready($signal.ready)"
  }

  for (i <- 0 until 2) {
    channel("CPU", s"cpuRequest_$i", s"cpu$i", new Request(p), i == 0)
    channel("RESULT", s"cpuResponse_$i", s"result$i", new Response(p), i == 0)
    channel("LINE", s"lineRequest_$i", s"line$i", new LineRequest(p.line), i == 0)
    channel("REPLY", s"lineResponse_$i", s"reply$i", new LineResponse(p.line), i == 0)
    channel("LINE", s"memoryRequest_$i", s"memory$i", new LineRequest(p.line), false)
    channel("REPLY", s"memoryResponse_$i", s"returned$i", new LineResponse(p.line), false)
  }
  ports += ".io_outstanding(control.outstanding)"
  Files.writeString(
    Paths.get("build/ram_config.svh"),
    "`ifndef RAM_CONFIG\n`define RAM_CONFIG\n" + defs.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(Paths.get("build/ram_ports.svh"), ports.mkString(",\n") + "\n")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Ddr(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/RamDdr", "--split-verilog")
  )
  val upstream = ports.filterNot(x => x.contains("io_memoryRequest_") || x.contains("io_memoryResponse_"))
  ports.clear(); ports ++= upstream; defs.clear()

  def observe(
    kind:   String,
    name:   String,
    signal: String,
    data:   Bundle
  ): Unit = {
    var offset = 0
    for ((field, value) <- data.elements.toSeq) {
      ports += s".io_${name}_bits_$field($signal.bits[$offset +: ${value.getWidth}])"; offset += value.getWidth
    }
    ports += s".io_${name}_valid($signal.valid)"
  }

  for (i <- 0 until 2) {
    observe("LINE", s"observedRequest_$i", s"memory$i", new LineRequest(p.line))
    observe("REPLY", s"observedResponse_$i", s"returned$i", new LineResponse(p.line))
  }
  val axi = memcore.bus.axi4.Params(addressBits = p.line.addressBits, dataBits = 128, idBits = 4)
  channel("A", "axi_aw", "aw", new memcore.bus.axi4.Address(axi))
  channel("A", "axi_ar", "ar", new memcore.bus.axi4.Address(axi), false)
  channel("W", "axi_w", "w", new memcore.bus.axi4.WriteData(axi))
  channel("B", "axi_b", "b", new memcore.bus.axi4.WriteResponse(axi))
  channel("R", "axi_r", "r", new memcore.bus.axi4.ReadData(axi))
  Files.writeString(
    Paths.get("build/ram_ddr_config.svh"),
    "`ifndef RAM_DDR_CONFIG\n`define RAM_DDR_CONFIG\n" + defs.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(Paths.get("build/ram_ddr_ports.svh"), ports.mkString(",\n") + "\n")

}
