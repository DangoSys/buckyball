package framework.acc.frontend

import sims.configs.SimParam
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import framework.acc.debug.FrontendDebugPort

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.boot.BootRom
import framework.acc.frontend.decoder.GlobalDecoder
import framework.acc.memdomain.bank_router.BankMapping
import framework.acc.frontend.scheduler.{Complete, Issue, MemoryIssue, Scheduler}
import framework.acc.memdomain.{Response => MemoryResponse}

class Command(dataBits: Int) extends Bundle {

  val raw_inst = UInt(32.W)
  val pc       = UInt(dataBits.W)
  val rs1Data  = UInt(dataBits.W)
  val rs2Data  = UInt(dataBits.W)
}

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
  val ownerId         = frontendParam.ownerId

  @public
  val io = IO(new Bundle {

    val rocc = new RoCCIO(commandDataBits, permissionParam)

    // Request payloads leave Frontend; each ready signal returns from its domain.
    val ballRequest    = Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam))
    // Completion payloads enter Frontend; Frontend drives their ready signals.
    val ballCompletion = Flipped(Decoupled(new MemoryResponse(frontendParam)))

    // Bank-version update request to MemDomain; its ready input gates the update.
    val mappingUpdate    = Decoupled(new BankMapping(frontendParam))
    val memoryRequest    = Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam))
    val memoryCompletion = Flipped(Decoupled(new MemoryResponse(frontendParam)))

    val gpRequest    = Option.when(gpParam.enable)(Decoupled(new Issue(frontendParam, memParam)))
    val gpCompletion = Option.when(gpParam.enable)(Flipped(Decoupled(new Complete(frontendParam))))

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
