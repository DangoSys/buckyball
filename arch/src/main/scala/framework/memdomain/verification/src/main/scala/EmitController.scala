package framework.memdomain.verification

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Path}
import framework.system.core.rocket.CpuParams
import framework.top.GlobalConfig

object EmitController {

  def apply(
    output:             Path,
    b:                  GlobalConfig,
    signatures:         Seq[BigInt],
    firtoolOptions:     Array[String]
  )(
    implicit cpuParams: CpuParams
  ): Unit = {
    var interface: Data = null
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      { val m = new ControllerSystem(b, signatures); interface = m.io; m },
      args = Array("--target-dir", output.resolve("ControllerSystem").toString, "--split-verilog"),
      firtoolOpts = firtoolOptions
    )
    def leaves(data: Data, name: String): Seq[(String, Data)] = data match {
      case bundle: Record => bundle.elements.toSeq.flatMap { case (field, value) => leaves(value, s"${name}_$field") }
      case vec:    Vec[_] => vec.zipWithIndex.flatMap { case (value, i) => leaves(value, s"${name}_$i") }.toSeq
      case value => Seq(name -> value)
    }
    val fields = leaves(interface, "io")
    Files.writeString(
      output.resolve("controller_system_signals.svh"),
      fields.map {
        case (name, data) => s"logic [${data.getWidth - 1}:0] $name;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("controller_system_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ fields.map { case (name, _) => s".$name($name)" }).mkString(
        ",\n"
      ) + "\n"
    )
    Files.writeString(
      output.resolve("controller_system_init.svh"),
      fields.collect {
        case (name, data) if DataMirror.directionOf(data) == ActualDirection.Input => s"$name = '0;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("controller_system_clocking.svh"),
      "clocking sample @(posedge clock);\ndefault input #1step;\ninput reset;\n" +
        fields.map {
          case (name, _) => s"input $name;"
        }.mkString("\n") + "\nendclocking\n"
    )
    Files.writeString(
      output.resolve("controller_system_config.svh"),
      s"`define CONTROLLER_SIGNATURE 64'h${signatures.head.toString(16)}\n"
    )
  }

}
