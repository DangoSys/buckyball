package framework.ant

import java.nio.file.{Files, Paths}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.spm

object Emit extends App {
  val p      = Params(256, spm.Params(0x10000, 256), spm.Params(0x20000, 256))
  val output = Paths.get("src/main/scala/framework/system/core/ant/build")
  Files.createDirectories(output)
  Files.writeString(
    output.resolve("config.svh"),
    s"`define TLS_BASE 64'h${p.data.base.toString(16)}\n" +
      s"`define TSS_BASE 64'h${p.shared.base.toString(16)}\n" +
      s"`define TLS_END 64'h${(p.data.base + p.data.bytes).toString(16)}\n"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Execution(p, contexts = 2),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Execution").toString, "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Control(p, contexts = 2),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Control").toString, "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new ServiceVerification(p, contexts = 2),
    firtoolOpts = args,
    args = Array("--target-dir", output.resolve("Service").toString, "--split-verilog")
  )
}

@instantiable
class ServiceVerification(p: Params, contexts: Int) extends Module {

  @public val io = IO(new Bundle {
    val request    = Flipped(Decoupled(new ControlRequest))
    val reply      = Decoupled(UInt(64.W))
    val signatures = Input(Vec(contexts, UInt(64.W)))
    val online     = Input(Vec(contexts, Bool()))
    val command    = Vec(contexts, Decoupled(new Command(p)))
    val response   = Vec(contexts, Flipped(Decoupled(new Response(p))))
    val npuDrained = Input(Vec(contexts, Bool()))
    val cancelNpu  = Output(Vec(contexts, Bool()))
    val launched   = Output(Vec(contexts, Valid(new Start(p))))
    val retired    = Output(Vec(contexts, Valid(new Retire)))
    val idle       = Output(Bool())
  })

  val service = Instantiate(new Service(p, contexts))
  service.io.request <> io.request
  io.reply <> service.io.reply
  service.io.signatures := io.signatures
  service.io.online     := io.online
  io.launched           := service.io.launched
  io.retired            := service.io.retired
  io.idle               := service.io.idle
  for (i <- 0 until contexts) {
    val execution = Instantiate(new LocalExecution(p))
    execution.io.local <> service.io.locals(i)
    execution.io.npuDrained := io.npuDrained(i)
    io.command(i) <> execution.io.command
    execution.io.response <> io.response(i)
    io.cancelNpu(i)         := service.io.locals(i).cancel
  }
}
