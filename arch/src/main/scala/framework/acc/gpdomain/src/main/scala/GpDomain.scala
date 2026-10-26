package framework.acc.gpdomain

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.scheduler.{Complete, Issue}
import framework.acc.memdomain.{BankLayout, BankLayoutRequest, BankPort}
import framework.acc.memdomain.toddr.DmaStatus
import framework.acc.gpdomain.execution.{Execution, KernelCompletion}
import framework.acc.gpdomain.loader.Loader
import framework.acc.gpdomain.memory.Memory
import framework.acc.balldomain.KernelBallPort
import framework.acc.gpdomain.decoder.Decoder
import framework.acc.gpdomain.loader.ImageLoad

class ImagePort extends Bundle {
  val start    = Decoupled(new ImageLoad)
  val data     = Decoupled(UInt(32.W))
  val terminal = Decoupled(new DmaStatus)
  val complete = Flipped(Decoupled(new KernelCompletion))
  val release  = Valid(Bool())
}

@instantiable
class GpDomain(val frontendParam: FrontendParam, val memParam: MemDomainParam, val gpParam: GpDomainParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val ownerId         = frontendParam.ownerId
  require(gpParam.enable)

  @public
  val io = IO(new Bundle {
    val layoutRequest  = Decoupled(new BankLayoutRequest(frontendParam))
    val layoutResponse = Flipped(Decoupled(new BankLayout(memParam)))
    val command        = Flipped(Decoupled(new Issue(frontendParam, memParam)))
    val complete       = Decoupled(new Complete(frontendParam))
    val image          = Flipped(new ImagePort)
    val banks          = Vec(gpParam.memory.memoryPorts, new BankPort(frontendParam, memParam))
    val ball           = new KernelBallPort(frontendParam, memParam)
  })

  val loader:    Instance[Loader]    = Instantiate(new Loader(frontendParam, memParam, gpParam))
  val execution: Instance[Execution] = Instantiate(new Execution(frontendParam, memParam, gpParam))
  val memory:    Instance[Memory]    = Instantiate(new Memory(frontendParam, memParam, gpParam))
  val decoder:   Instance[Decoder]   = Instantiate(new Decoder(frontendParam, memParam))

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
