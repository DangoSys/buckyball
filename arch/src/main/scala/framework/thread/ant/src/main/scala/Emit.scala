package framework.thread.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.Instantiate
import framework.core.controller.configs.TasksParam
import framework.core.controller.task.{Control, Service, TaskRequest}
import framework.thread.ant.{Completion, Retire}
import framework.acc.frontend.RoCCIO
import framework.core.tss.{Store => TssStore}
import framework.core.tss.configs.{Params => TssParam}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.preflight.PmpEntry
import framework.ip.memory.spm
import java.nio.file.{Files, Paths}

/** Direct task/transfer verification around real Ants and their shared storage. */
class AntVerification(p: TasksParam) extends Module {
  require(p.locals.size == 2 && p.locals.distinct.size == 1)
  val local = p.locals.head

  val io = IO(new Bundle {
    val inUse       = Input(Bool())
    val start       = Vec(2, Flipped(Decoupled(new Launch(local, p.permission))))
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
    ant.io.task.inUse        := io.inUse
    ant.io.task.start.valid  := io.start(i).valid && !shared.io.busy(i) && !shared.io.hostBusy
    ant.io.task.start.bits   := io.start(i).bits
    io.start(i).ready        := ant.io.task.start.ready && !shared.io.busy(i) && !shared.io.hostBusy
    io.result(i).valid       := ant.io.task.result.valid && !shared.io.busy(i)
    io.result(i).bits        := ant.io.task.result.bits
    ant.io.task.result.ready := io.result(i).ready && !shared.io.busy(i)
    ant.io.task.transfer <> io.transfer(i)
    io.transferred(i) <> ant.io.task.transferred
    ant.io.task.cancel       := io.cancel(i)
    io.running(i)            := ant.io.task.running
    io.rocc(i) <> ant.io.rocc
    io.retired(i)            := ant.io.trace
    shared.io.clients(i) <> ant.io.shared
  }
}

/** Production management service connected to real Ant/TSS instances. */
class ServiceVerification(p: TasksParam) extends Module {

  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new TaskRequest(p.permission)))
    val reply   = Decoupled(UInt(64.W))
    val rocc    = Vec(p.count, Flipped(new RoCCIO(64, p.permission)))
    val retired = Output(Vec(p.count, Valid(new Retire)))
  })

  val service = Instantiate(new Service(p))
  val shared  = Instantiate(new TssStore(p.tss))
  service.io.request <> io.request
  io.reply <> service.io.reply
  shared.io.host <> service.io.tss.host
  shared.io.inUse         := service.io.tss.inUse
  shared.io.cancel        := service.io.tss.cancel
  service.io.tss.busy     := shared.io.busy
  service.io.tss.hostBusy := shared.io.hostBusy
  for (i <- 0 until p.count) {
    val ant = Instantiate(new Ant(p.locals(i), p.permission))
    ant.io.task <> service.io.tasks(i)
    shared.io.clients(i) <> ant.io.shared
    io.rocc(i) <> ant.io.rocc
    io.retired(i) := ant.io.trace
  }
}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  def memory(value: ujson.Value): spm.Params =
    spm.Params(BigInt(value("base").str), value("bytes").num.toInt, value("dataBits").num.toInt)
  val shared     = memory(values("tss"))
  val local      =
    configs.AntParam(values("codeBytes").num.toInt, memory(values("tls")), shared, values("taskBits").num.toInt)
  val signatures = values("signatures").arr.map(value => BigInt(value.str)).toSeq
  require(signatures.size == 2, "Ant verification uses two hardware threads")

  val p = TasksParam(
    Seq(local, local),
    signatures,
    TssParam(shared, 2),
    PmpParams(values("paddrBits").num.toInt, values("pmps").num.toInt)
  )

  require(local.taskBits == 32 && shared.dataBits == 128)
  require(local.data.base == 0x10000 && shared.base == 0x20000, "Shared-program fixture address encoding mismatch")
  // PMP entries connect to flat per-context vectors so benches compare whole contexts.
  val entry = new PmpEntry(p.permission)

  val fields = Seq(
    "cfg_l"   -> entry.cfg.l.getWidth,
    "cfg_res" -> entry.cfg.res.getWidth,
    "cfg_a"   -> entry.cfg.a.getWidth,
    "cfg_x"   -> entry.cfg.x.getWidth,
    "cfg_w"   -> entry.cfg.w.getWidth,
    "cfg_r"   -> entry.cfg.r.getWidth,
    "addr"    -> entry.addr.getWidth,
    "mask"    -> entry.mask.getWidth
  )

  val entryBits = fields.map(_._2).sum

  def pmp(port: String, signal: String): Seq[String] = for {
    i                        <- 0 until p.permission.entries
    ((field, width), offset) <- fields.zip(fields.scanLeft(0)(_ + _._2))
  } yield s", .${port}_pmp_${i}_$field($signal[${i * entryBits + offset + width - 1}:${i * entryBits + offset}])"

  val directory = Paths.get(output)
  Files.createDirectories(directory)
  Files.writeString(
    directory.resolve("config.svh"),
    s"""`define LOCAL_BASE 64'h${local.data.base.toString(16)}
`define LOCAL_BYTES ${local.data.bytes}
`define TLS_BASE 64'h${local.data.base.toString(16)}
`define TLS_END 64'h${(local.data.base + local.data.bytes).toString(16)}
`define TSS_BASE 64'h${shared.base.toString(16)}
`define TSS_SCRATCH 64'h${(shared.base + shared.bytes - 32).toString(16)}
`define CODE_BYTES ${local.codeBytes}
`define TLS_BYTES ${local.data.bytes}
`define TSS_BYTES ${shared.bytes}
`define SIGNATURE_0 64'h${p.signatures(0).toString(16)}
`define SIGNATURE_1 64'h${p.signatures(1).toString(16)}
`define PMP_BITS ${p.permission.entries * entryBits}
"""
  )
  Files.writeString(
    directory.resolve("control_pmp.svh"),
    (pmp("io_request_bits_context", "request_pmp") ++
      (0 until 2).flatMap(c => pmp(s"io_start_${c}_bits_context", s"launch_pmp[$c]"))).mkString("\n")
  )
  Files.writeString(
    directory.resolve("ant_pmp.svh"),
    (0 until 2).flatMap(c =>
      pmp(s"io_start_${c}_bits_context", s"start_pmp[$c]") ++ pmp(s"io_rocc_${c}_cmd_bits", s"command_pmp[$c]")
    ).mkString("\n")
  )
  Files.writeString(
    directory.resolve("service_pmp.svh"),
    (pmp("io_request_bits_context", "request_pmp") ++
      (0 until 2).flatMap(c => pmp(s"io_rocc_${c}_cmd_bits", s"command_pmp[$c]"))).mkString("\n")
  )
  Seq[(String, () => RawModule)](
    "Store"   -> (() => new LocalStore(local, 1)),
    "Ant"     -> (() => new AntVerification(p)),
    "Control" -> (() => new Control(p)),
    "Service" -> (() => new ServiceVerification(p))
  ).foreach { case (name, top) =>
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      top(),
      Array("--target-dir", directory.resolve(name).toString, "--split-verilog"),
      firtoolOpts = Array("--disable-all-randomization")
    )
  }
}
