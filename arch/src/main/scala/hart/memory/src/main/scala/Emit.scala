package framework.hart.memory

import framework.tlp.memory.Dataflow

import chisel3._
import chisel3.reflect.DataMirror
import framework.tlp.shm.{MeshSharedMemParams, MeshThreadAttachment}
import framework.generic.bus.axi4
import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))

  val p = MeshSharedMemParams(
    rows = values("rows").num.toInt,
    cols = values("cols").num.toInt,
    entriesPerBank = values("entriesPerBank").num.toInt,
    dataBits = values("dataBits").num.toInt,
    tagBits = values("tagBits").num.toInt,
    threads = values("threads").arr.toSeq.map(t => MeshThreadAttachment(t("bank").num.toInt, t("ports").num.toInt)),
    banksPerNode = values("banksPerNode").num.toInt,
    maxInFlight = values("maxInFlight").num.toInt
  )

  val bus         = axi4.InterconnectParam(axi4.Params(44, p.dataBits, 4), p.threads.size, Some(values("slots").num.toInt))
  val definitions = scala.collection.mutable.ArrayBuffer[String]()
  val bindings    = scala.collection.mutable.ArrayBuffer[String]()
  var inputOffset = 0; var outputOffset = 0

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
      val dut = new Dataflow(p, bus)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define DF_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define DF_${direction}_${name.toUpperCase}_WIDTH ${field.getWidth}"
        bindings += s".io_$name(bus.$signal[$offset +: ${field.getWidth}])"
        if (input) inputOffset += field.getWidth else outputOffset += field.getWidth
      }
      dut
    },
    args = Array("--target-dir", s"$directory/Dataflow", "--split-verilog"),
    firtoolOpts = Array("-lowering-options=disallowLocalVariables")
  )
  definitions += s"`define DF_INPUT_BITS $inputOffset"
  definitions += s"`define DF_OUTPUT_BITS $outputOffset"
  definitions += s"`define DF_BANKS ${p.bankCount}"
  definitions += s"`define DF_THREADS ${p.threads.size}"
  Files.writeString(Paths.get(directory, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "ports.svh"), bindings.mkString(",\n") + "\n")
}
