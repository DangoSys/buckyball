package framework.thread.ant

import buckyball.config.Design
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.Instantiate
import framework.core.controller.configs.TasksParam
import framework.core.controller.task.{Control, Service}
import framework.thread.ant.{Completion, Retire, Start}
import framework.acc.frontend.RoCCIO
import framework.core.tss.{Store => TssStore}
import framework.core.tss.configs.{Params => TssParam}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.device.cbus.ControlRequest
import framework.ip.memory.spm
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._

/** Direct task/transfer verification around real Ants and their shared storage. */
class AntVerification(p: TasksParam) extends Module {
  require(p.locals.size == 2 && p.locals.distinct.size == 1)
  val local = p.locals.head

  val io = IO(new Bundle {
    val inUse       = Input(Bool())
    val start       = Vec(2, Flipped(Decoupled(new Start(local))))
    val result      = Vec(2, Decoupled(new Completion(local)))
    val transfer    = Vec(2, Flipped(Decoupled(new Transfer)))
    val transferred = Vec(2, Decoupled(Bool()))
    val cancel      = Input(Vec(2, Bool()))
    val running     = Output(Vec(2, Bool()))
    val rocc        = Vec(2, Flipped(new RoCCIO(64, p.permission)))
    val retired     = Output(Vec(2, Valid(new Retire)))
    val host        = Flipped(new spm.Port(p.tss.memory))
  })

  val ants   = Seq.fill(2)(Instantiate(new Ant(local, p.permission)))
  val shared = Instantiate(new TssStore(p.tss))
  shared.io.host <> io.host
  shared.io.inUse  := io.inUse || ants.map(_.io.task.storageBusy).reduce(_ || _)
  shared.io.cancel := io.cancel
  for ((ant, i) <- ants.zipWithIndex) {
    ant.io.task.inUse                 := io.inUse
    ant.io.task.start.valid           := io.start(i).valid && !shared.io.busy(i) && !shared.io.hostBusy
    ant.io.task.start.bits.descriptor := io.start(i).bits
    ant.io.task.start.bits.context    := 0.U.asTypeOf(ant.io.task.start.bits.context)
    io.start(i).ready                 := ant.io.task.start.ready && !shared.io.busy(i) && !shared.io.hostBusy
    io.result(i).valid                := ant.io.task.result.valid && !shared.io.busy(i)
    io.result(i).bits                 := ant.io.task.result.bits
    ant.io.task.result.ready          := io.result(i).ready && !shared.io.busy(i)
    ant.io.task.transfer <> io.transfer(i)
    io.transferred(i) <> ant.io.task.transferred
    ant.io.task.cancel                := io.cancel(i)
    io.running(i)                     := ant.io.task.running
    io.rocc(i) <> ant.io.rocc
    io.retired(i)                     := ant.io.trace
    shared.io.clients(i) <> ant.io.shared
  }
}

/** Production management service connected to real Ant/TSS instances. */
class ServiceVerification(p: TasksParam) extends Module {

  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new ControlRequest))
    val reply   = Decoupled(UInt(64.W))
    val rocc    = Vec(p.count, Flipped(new RoCCIO(64, p.permission)))
    val retired = Output(Vec(p.count, Valid(new Retire)))
  })

  val service = Instantiate(new Service(p))
  val shared  = Instantiate(new TssStore(p.tss))
  service.io.request.valid        := io.request.valid
  service.io.request.bits.control := io.request.bits
  service.io.request.bits.context := 0.U.asTypeOf(service.io.request.bits.context)
  io.request.ready                := service.io.request.ready
  io.reply <> service.io.reply
  shared.io.host <> service.io.tss.host
  shared.io.inUse                 := service.io.tss.inUse
  shared.io.cancel                := service.io.tss.cancel
  service.io.tss.busy             := shared.io.busy
  service.io.tss.hostBusy         := shared.io.hostBusy
  for (i <- 0 until p.count) {
    val ant = Instantiate(new Ant(p.locals(i), p.permission))
    ant.io.task <> service.io.tasks(i)
    shared.io.clients(i) <> ant.io.shared
    io.rocc(i) <> ant.io.rocc
    io.retired(i) := ant.io.trace
  }
}

object Emit extends App {
  require(args.length == 4 && args(0) == "--config" && args(2) == "--core")
  val design  = Design.parseFrom(Files.readAllBytes(Paths.get(args(1))))
  val core    = design.getDevice.getCoresList.asScala.find(_.getHartId == args(3).toInt).get
  val shared  = core.getTss
  val memory  = spm.Params(BigInt(java.lang.Long.toUnsignedString(shared.getBase)), shared.getBytes, shared.getDataBits)
  require(core.getThreadIndicesCount >= 2, "Ant verification requires two configured hardware threads")
  val threads = core.getThreadIndicesList.asScala.take(2).map(i => design.getThreads(i.intValue)).toSeq
  val locals  = threads.map(t => configs.AntParam.fromProto(t.getCpu.getAnt, memory))

  val p = TasksParam(
    locals,
    threads.map(t => BigInt(java.lang.Long.toUnsignedString(t.getSignature))),
    TssParam(memory, 2),
    PmpParams(core.getParam.getPaddrBits, core.getParam.getNPmps)
  )

  require(locals.distinct.size == 1 && locals.head.taskBits == 32 && memory.dataBits == 128)
  require(locals.head.data.base == 0x10000 && memory.base == 0x20000, "Shared-program fixture address encoding mismatch")
  val directory = Paths.get("build")
  Files.createDirectories(directory)
  Files.writeString(
    directory.resolve("config.svh"),
    s"""`define LOCAL_BASE 64'h${locals.head.data.base.toString(16)}
`define LOCAL_BYTES ${locals.head.data.bytes}
`define TLS_BASE 64'h${locals.head.data.base.toString(16)}
`define TLS_END 64'h${(locals.head.data.base + locals.head.data.bytes).toString(16)}
`define TSS_BASE 64'h${memory.base.toString(16)}
`define TSS_SCRATCH 64'h${(memory.base + memory.bytes - 32).toString(16)}
`define CODE_BYTES ${locals.head.codeBytes}
`define TLS_BYTES ${locals.head.data.bytes}
`define TSS_BYTES ${memory.bytes}
`define SIGNATURE_0 64'h${p.signatures(0).toString(16)}
`define SIGNATURE_1 64'h${p.signatures(1).toString(16)}
"""
  )
  val fields    = Seq("cfg_l", "cfg_res", "cfg_a", "cfg_x", "cfg_w", "cfg_r", "addr", "mask")
  Files.writeString(
    directory.resolve("control_pmp.svh"),
    (for {
      i     <- 0 until p.permission.entries
      field <- fields
    } yield s", .io_request_bits_context_pmp_${i}_$field('0)").mkString("\n")
  )
  Seq[(String, () => RawModule)](
    "Store"   -> (() => new LocalStore(locals.head, 1)),
    "Ant"     -> (() => new AntVerification(p)),
    "Control" -> (() => new Control(p)),
    "Service" -> (() => new ServiceVerification(p))
  ).foreach { case (name, top) =>
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      top(),
      Array("--target-dir", s"build/$name", "--split-verilog"),
      firtoolOpts = Array("--disable-all-randomization")
    )
  }
}
