package framework.memdomain.verification

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Path}
import framework.system.memory.Memory
import memcore.memory.ddr.{Params => DdrParams}

object EmitMemory {

  def apply(output: Path, firtoolOptions: Array[String]): Unit = {
    val ddr = DdrParams()
    var interface: Data = null
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      { val m = new Memory(ddr, dmaMasters = 1); interface = m.io; m },
      args = Array("--target-dir", output.resolve("Memory").toString, "--split-verilog"),
      firtoolOpts = firtoolOptions
    )
    def leaves(data: Data, name: String): Seq[(String, Data)] = data match {
      case bundle: Record => bundle.elements.toSeq.flatMap { case (field, value) => leaves(value, s"${name}_$field") }
      case vec:    Vec[_] => vec.zipWithIndex.flatMap { case (value, i) => leaves(value, s"${name}_$i") }.toSeq
      case value => Seq(name -> value)
    }
    val fields = leaves(interface, "io")
    Files.writeString(
      output.resolve("memory_system_signals.svh"),
      fields.map {
        case (name, data) => s"logic [${data.getWidth - 1}:0] $name;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("memory_system_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ fields.map { case (name, _) => s".$name($name)" }).mkString(
        ",\n"
      ) + "\n"
    )
    Files.writeString(
      output.resolve("memory_system_init.svh"),
      fields.collect {
        case (name, data) if DataMirror.directionOf(data) == ActualDirection.Input => s"$name = '0;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("memory_system_clocking.svh"),
      "clocking sample @(posedge clock);\ndefault input #1step;\ninput reset;\n" +
        "" +
        "input line_ready,reply_valid,reply_id,reply_data,reply_error;\n" +
        fields.map {
          case (name, _) => s"input $name;"
        }.mkString("\n") + "\nendclocking\n"
    )
  }

}
