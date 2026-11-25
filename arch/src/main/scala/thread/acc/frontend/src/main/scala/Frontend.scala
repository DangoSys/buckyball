package framework.thread.acc.frontend

import framework.thread.acc.configs.BallLink

import framework.generic.contract.blink.BlinkParam

import framework.thread.acc.debug.FrontendDebugPort

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.acc.frontend.boot.BootRom
import framework.thread.acc.frontend.decoder.GlobalDecoder
import framework.thread.acc.memdomain.bank_router.BankMappingIO
import framework.thread.acc.frontend.scheduler.Scheduler
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.RoCCIO
import framework.generic.contract.configs.BallDomainParam
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.frontend.scheduler.Complete
import framework.generic.contract.frontend.scheduler.Issue
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.memdomain.{Response => MemoryResponse}

/** Frontend composition; boundary ports connect directly to their owning child IP. */
@instantiable
class Frontend(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val ballParam:       BallDomainParam,
  val permissionParam: PmpParams,
  val simp:            SimParam)
    extends Module {
  val diffTest        = simp.diffTest
  val commandDataBits = frontendParam.commandDataBits

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId = Input(UInt(64.W))

    val rocc = new RoCCIO(commandDataBits, permissionParam)

    // Request payloads leave Frontend; each ready signal returns from its domain.
    val ballRequest    = Decoupled(new MemoryIssue(BallLink.blink(frontendParam, memParam), permissionParam))
    // Completion payloads enter Frontend; Frontend drives their ready signals.
    val ballCompletion = Flipped(Decoupled(new MemoryResponse(log2Up(frontendParam.rob_entries))))

    // Bank-version update request to MemDomain; its ready input gates the update.
    val mappingUpdate    = new BankMappingIO(frontendParam)
    val memoryRequest    = Decoupled(new MemoryIssue(BallLink.blink(frontendParam, memParam), permissionParam))
    val memoryCompletion = Flipped(Decoupled(new MemoryResponse(log2Up(frontendParam.rob_entries))))

    val gpRequest    = Option.when(gpParam.enable)(Decoupled(new Issue(BallLink.blink(frontendParam, memParam))))
    val gpCompletion = Option.when(gpParam.enable)(Flipped(Decoupled(new Complete(log2Up(frontendParam.rob_entries)))))

    val barrier = new Bundle {
      val arrive  = Output(Bool())
      val release = Input(Bool())
    }

    val debug = new FrontendDebugPort(frontendParam, simp)
  })

  val ingress:   Instance[Ingress]       = Instantiate(new Ingress(permissionParam, frontendParam))
  val boot:      Instance[BootRom]       = Instantiate(new BootRom(ballParam.balls.map(_.id), permissionParam, frontendParam))
  val gDecoder:  Instance[GlobalDecoder] =
    Instantiate(new GlobalDecoder(frontendParam, memParam, gpParam, permissionParam))
  val scheduler: Instance[Scheduler]     =
    Instantiate(new Scheduler(frontendParam, memParam, gpParam, permissionParam, simp))
  scheduler.io.ownerId := io.ownerId

  io.rocc <> ingress.io.rocc
  boot.io.command <> ingress.io.command
  ingress.io.fault      := scheduler.io.fault
  ingress.io.idle       := boot.io.idle
  gDecoder.io.command <> boot.io.cmd
  scheduler.io.admission.command <> gDecoder.io.decoded
  boot.io.schedulerIdle := scheduler.io.admission.idle

  io.ballRequest <> scheduler.io.ballRequest
  scheduler.io.ballCompletion <> io.ballCompletion
  io.memoryRequest <> scheduler.io.memoryRequest
  scheduler.io.memoryCompletion <> io.memoryCompletion
  io.gpRequest.zip(scheduler.io.gpRequest).foreach { case (port, source) => port <> source }
  io.gpCompletion.zip(scheduler.io.gpCompletion).foreach { case (port, sink) => sink <> port }
  io.debug.trace := scheduler.io.traceInstructions
  io.debug.control.zip(scheduler.io.debug).foreach { case (port, source) => port <> source }
  io.mappingUpdate <> scheduler.io.mappingUpdate

  io.barrier.arrive            := scheduler.io.barrier.arrive
  scheduler.io.barrier.release := io.barrier.release

}
