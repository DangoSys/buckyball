package memcore.memory.mmu

import java.nio.file.{Files, Paths}
import memcore.bus.chi.Params

object Emit extends App {
  val p                 = Params()
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Walker(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Walker", "--split-verilog")
  )
  val walkerDefinitions = scala.collection.mutable.ArrayBuffer[String]()
  val walkerPorts       = scala.collection.mutable.ArrayBuffer[String]()

  def walkerChannel(
    kind:   String,
    port:   String,
    signal: String,
    fs:     Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((field, width) <- fs) {
      walkerDefinitions += s"`define WALK_${kind}_${field.toUpperCase}_OFFSET $offset"
      walkerDefinitions += s"`define WALK_${kind}_${field.toUpperCase}_WIDTH $width"
      walkerPorts += s".io_${port}_bits_$field($signal.bits[$offset +: $width])"
      offset += width
    }
    walkerDefinitions += s"`define WALK_${kind}_WIDTH $offset"
    walkerPorts += s".io_${port}_valid($signal.valid)"
    walkerPorts += s".io_${port}_ready($signal.ready)"
  }

  walkerChannel(
    "REQ",
    "req",
    "source_if",
    Seq("vaddr" -> 64, "write" -> 1, "execute" -> 1, "privilege" -> 2, "sum" -> 1, "mxr" -> 1)
  )
  walkerChannel(
    "RESP",
    "resp",
    "sink_if",
    Seq("paddr" -> p.addressBits, "pageFault" -> 1, "accessFault" -> 1, "level" -> 2)
  )
  walkerChannel(
    "ACCESS",
    "access",
    "access_if",
    Seq("addr" -> p.addressBits, "write" -> 1, "data" -> 64, "mask" -> 8, "atomic" -> 4, "atomicWord" -> 1)
  )
  walkerChannel("RESULT", "result", "result_if", Seq("data" -> 64, "error" -> 1))
  Files.writeString(
    Paths.get("build/walker_config.svh"),
    "`ifndef WALKER_CONFIG_SVH\n`define WALKER_CONFIG_SVH\n" + walkerDefinitions.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(Paths.get("build/walker_ports.svh"), walkerPorts.mkString(",\n") + "\n")
}
