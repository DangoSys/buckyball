package framework.acc.memdomain.tomem

import chisel3._
import chisel3.util._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

/** Verification assembly includes the direct-accelerator translation and coherence adapters. */
class ToMemVerification(
  frontend:   FrontendParam,
  memParam:   MemDomainParam,
  gp:         GpDomainParam,
  transport:  configs.Params,
  permission: PmpParams)
    extends Module {
  val mover     = chisel3.experimental.hierarchy.Instantiate(new ToMem(frontend, memParam, gp, transport, permission))
  val ddr       = chisel3.experimental.hierarchy.Instantiate(new framework.acc.memdomain.toddr.ToDDR(transport, permission))
  val coherence = chisel3.experimental.hierarchy.Instantiate(new framework.acc.memdomain.tol1.ToL1(transport.interlock))

  val io = IO(new Bundle {
    val request     = Flipped(Decoupled(new framework.acc.memdomain.Request(frontend, memParam, permission)))
    val response    = Decoupled(new framework.acc.memdomain.Response(frontend))
    val bankRead    = new framework.acc.memdomain.BankPort(frontend, memParam)
    val bankWrite   = new framework.acc.memdomain.BankPort(frontend, memParam)
    val metadata    = Flipped(new framework.acc.balldomain.mmio_bank.MetadataWrite(memParam))
    val image       = Option.when(gp.enable)(new framework.acc.gpdomain.ImagePort)
    val queryVbank  = Output(UInt(frontend.bank_id_len.W))
    val queryShared = Output(Bool())
    val queryGroups = Input(UInt(memParam.groupCountWidth.W))

    val memory = new Bundle {

      val control = new ControlPort(transport, permission) {
        val decisions = Input(Vec(3, new framework.acc.memdomain.toddr.DmaDecision(transport)))
        val queries   = Output(Vec(2, new framework.core.memory.access.preflight.MapQuery(transport.mapping)))
        val mappings  = Input(Vec(2, new framework.core.memory.access.preflight.MapResult(transport.mapping)))
        val coherence = new framework.acc.memdomain.tol1.Port(transport.interlock)
      }

      val readAddress  = Output(UInt(64.W))
      val writeAddress = Output(UInt(64.W))
      val axi          = new framework.ip.bus.axi4.Port(transport.axi)
    }

  })

  mover.io.request <> io.request
  io.response <> mover.io.response
  io.bankRead <> mover.io.bankRead
  io.bankWrite <> mover.io.bankWrite
  mover.io.metadata <> io.metadata
  io.image.zip(mover.io.image).foreach { case (port, source) => port <> source }
  io.queryVbank              := mover.io.queryVbank
  io.queryShared             := mover.io.queryShared
  mover.io.queryGroups       := io.queryGroups
  io.memory.control.read     := mover.io.memory.control.read
  io.memory.control.write    := mover.io.memory.control.write
  io.memory.control.finished := mover.io.memory.control.finished
  mover.io.memory.control.complete <> io.memory.control.complete
  ddr.io.dma <> mover.io.dma
  ddr.io.decisions           := io.memory.control.decisions
  io.memory.control.queries  := ddr.io.queries
  ddr.io.mappings            := io.memory.control.mappings
  io.memory.axi <> ddr.io.axi
  io.memory.readAddress      := ddr.io.readAddress
  io.memory.writeAddress     := ddr.io.writeAddress
  coherence.io <> io.memory.control.coherence
}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val frontend                                             = upickle.default.read[FrontendParam](values("frontend"))
  val memory                                               = upickle.default.read[MemDomainParam](values("memory"))
  val gp                                                   = upickle.default.read[GpDomainParam](values("gp"))
  val permission                                           = upickle.default.read[PmpParams](values("permission"))
  val transport                                            = upickle.default.read[configs.Params](values("transport"))

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define TM_BANKS ${memory.virtualBankCount + 2}",
    s"`define TM_ROWS ${memory.bankEntries}",
    s"`define TM_GROUPS ${memory.bankNum}",
    s"`define TM_DATA_BITS ${memory.bankWidth}"
  )

  val binding     = scala.collection.mutable.ArrayBuffer[String]()
  var inputOffset = 0; var outputOffset = 0
  Files.createDirectories(Paths.get(directory))

  def leaves(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        leaves(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => leaves(field, s"${path}_$index") }
    case field => if (field.getWidth > 0) Seq(path -> field) else Seq.empty
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new ToMemVerification(frontend, memory, gp, transport, permission)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal              = if (input) "inputs" else "outputs";
        val offset    = if (input) inputOffset else outputOffset; val width = field.getWidth
        definitions += s"`define TM_${direction}_${name.toUpperCase}_OFFSET $offset";
        definitions += s"`define TM_${direction}_${name.toUpperCase}_WIDTH $width"
        binding += s".io_$name(bus.$signal[$offset +: $width])";
        if (input) inputOffset += width else outputOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/ToMem", "--split-verilog")
  )
  definitions += s"`define TM_INPUT_BITS $inputOffset"; definitions += s"`define TM_OUTPUT_BITS $outputOffset"
  Files.writeString(Paths.get(directory, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "ports.svh"), binding.mkString(",\n") + "\n")
}
