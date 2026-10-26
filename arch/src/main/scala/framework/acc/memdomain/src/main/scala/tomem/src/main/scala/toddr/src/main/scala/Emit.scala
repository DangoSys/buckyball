package framework.acc.memdomain.toddr

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.core.memory.access.preflight.configs.PmpParams

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val transport                                            = upickle.default.read[TransportParam](values("transport"))
  val permission                                           = upickle.default.read[PmpParams](values("permission"))
  Files.createDirectories(Paths.get(directory))

  def leaves(data: Data, prefix: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) => leaves(field, s"${prefix}_$name") }
    case vector: Vec[_] => vector.zipWithIndex.flatMap { case (field, index) =>
        leaves(field, s"${prefix}_$index")
      }.toSeq
    case field => if (field.getWidth > 0) Seq(prefix -> field) else Seq.empty
  }

  for ((name, top) <- Seq("toddr" -> "ToDDR", "read_dma" -> "ReadDma", "write_dma" -> "WriteDma")) {
    var interface: Data = null
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      name match {
        case "toddr"     =>
          val m =
            new ToDDR(transport, permission);
          interface = m.io; m
        case "read_dma"  => val m = new ReadDma(transport); interface = m.io; m
        case "write_dma" =>
          val m =
            new WriteDma(transport, permission);
          interface = m.io; m
      },
      args = Array("--target-dir", s"$directory/$top", "--split-verilog")
    )
    val fields = leaves(interface, "io")
    Files.writeString(
      Paths.get(s"$directory/${name}_signals.svh"),
      fields.map { case (n, d) => s"logic [${d.getWidth - 1}:0] $n;" }.mkString("\n") + "\n"
    )
    Files.writeString(
      Paths.get(s"$directory/${name}_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ fields.map { case (n, _) => s".$n($n)" }).mkString(",\n") + "\n"
    )
    Files.writeString(
      Paths.get(s"$directory/${name}_binding.svh"),
      (Seq(".clock(clock)", ".reset(bus.reset)") ++ fields.map { case (n, _) => s".$n(bus.$n)" }).mkString(",\n") + "\n"
    )
    Files.writeString(
      Paths.get(s"$directory/${name}_init.svh"),
      fields.collect { case (n, d) if DataMirror.directionOf(d) == ActualDirection.Input => s"$n = '0;" }.mkString(
        "\n"
      ) + "\n"
    )
    Files.writeString(
      Paths.get(s"$directory/${name}_clocking.svh"),
      "clocking sample @(posedge clock);\ndefault input #1step;\ninput reset;\n" + fields.map { case (n, _) =>
        s"input $n;"
      }.mkString("\n") + "\nendclocking\n"
    )
  }
}
