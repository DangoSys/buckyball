package framework.thread.acc.gpdomain

import framework.thread.acc.configs.BallLink

import framework.generic.contract.blink.BlinkParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.acc.gpdomain.execution.{Execution, OperatorCompletion}
import framework.thread.acc.gpdomain.loader.Loader
import framework.thread.acc.gpdomain.memory.Memory
import framework.thread.acc.gpdomain.decoder.Decoder
import framework.thread.acc.gpdomain.loader.ImageLoad
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.frontend.scheduler.Complete
import framework.generic.contract.frontend.scheduler.Issue
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.toddr.DmaStatus
import framework.generic.contract.ball.OperatorBallPort

class ImagePort extends Bundle {
  val start    = Decoupled(new ImageLoad)
  val data     = Decoupled(UInt(32.W))
  val terminal = Decoupled(new DmaStatus)
  val complete = Flipped(Decoupled(new OperatorCompletion))
  val release  = Valid(Bool())
}

@instantiable
class GpDomain(val frontendParam: FrontendParam, val memParam: MemDomainParam, val gpParam: GpDomainParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  require(gpParam.enable)

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId        = Input(UInt(64.W))
    val layoutRequest  = Decoupled(new BankLayoutRequest(frontendParam.bank_id_len))
    val layoutResponse = Flipped(Decoupled(new BankLayout(memParam.groupCountWidth)))
    val command        = Flipped(Decoupled(new Issue(BallLink.blink(frontendParam, memParam))))
    val complete       = Decoupled(new Complete(log2Up(frontendParam.rob_entries)))
    val image          = Flipped(new ImagePort)
    val banks          = Vec(gpParam.memory.memoryPorts, new BankPort(BallLink.blink(frontendParam, memParam)))
    val ball           = new OperatorBallPort(BallLink.blink(frontendParam, memParam))
  })

  val loader:    Instance[Loader]    = Instantiate(new Loader(frontendParam, memParam, gpParam))
  val execution: Instance[Execution] = Instantiate(new Execution(frontendParam, memParam, gpParam))
  val memory:    Instance[Memory]    = Instantiate(new Memory(frontendParam, memParam, gpParam))
  memory.io.ownerId := io.ownerId
  val decoder: Instance[Decoder] = Instantiate(new Decoder(frontendParam, memParam))

  decoder.io.issue <> io.command
  io.layoutRequest <> decoder.io.layoutRequest
  decoder.io.layoutResponse <> io.layoutResponse
  loader.io.cmdReq <> decoder.io.request

  execution.io.program <> loader.io.program
  execution.io.launch <> loader.io.launch
  loader.io.done <> execution.io.done
  loader.io.memory <> memory.io.loader

  for (port <- 0 until gpParam.memory.memoryPorts) {
    memory.io.request(port) <> execution.io.memoryRequest(port)
    execution.io.memoryResponse(port) <> memory.io.response(port)
    io.banks(port) <> memory.io.banks(port)
  }

  io.ball <> execution.io.ball

  loader.io.uploadStart <> io.image.start
  loader.io.image <> io.image.data
  loader.io.imageTerminal <> io.image.terminal
  io.image.complete <> loader.io.uploadComplete
  loader.io.release := io.image.release

  io.complete <> loader.io.cmdResp
}
