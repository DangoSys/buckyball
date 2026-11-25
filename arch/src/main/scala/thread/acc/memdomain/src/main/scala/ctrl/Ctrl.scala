package framework.thread.acc.memdomain

import framework.thread.acc.configs.BallLink

import framework.generic.contract.blink.BlinkParam

import framework.thread.acc.memdomain.LayoutConfig
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort

import framework.generic.contract.memdomain.MemoryContext

import framework.generic.arbiter.{Arbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.memdomain.toddr.DmaStatus
import framework.generic.contract.memdomain.Response

class Request(val frontendParam: FrontendParam, val memParam: MemDomainParam, val permissionParam: PmpParams)
    extends Bundle {
  val context = new MemoryContext(permissionParam)
  val instId  = UInt(64.W)
  val cmd     = new DecodedCommand(frontendParam, memParam)
  val rob_id  = UInt(log2Up(frontendParam.rob_entries).W)
}

@instantiable
class Ctrl(val frontendParam: FrontendParam, val memParam: MemDomainParam, val permissionParam: PmpParams)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId   = Input(UInt(64.W))
    val issue     = Flipped(Decoupled(new MemoryIssue(BallLink.blink(frontendParam, memParam), permissionParam)))
    val complete  = Decoupled(new Response(log2Up(frontendParam.rob_entries)))
    val request   = Decoupled(new Request(frontendParam, memParam, permissionParam))
    val response  = Flipped(Decoupled(new Response(log2Up(frontendParam.rob_entries))))
    val config    = Decoupled(new LayoutConfig(frontendParam, memParam))
    val clearBusy = Input(Bool())
    val mvover    = new MvoverPort
  })

  val decoder:  Instance[Decoder]  = Instantiate(new Decoder(frontendParam, memParam))
  val configer: Instance[Configer] = Instantiate(new Configer(frontendParam, memParam, permissionParam))
  configer.io.ownerId := io.ownerId

  val operation     = io.issue.bits.operation
  val isMvover      = operation.cmd.cmd.raw_inst(31, 25) === MvoverISA.Funct.U
  val mvoverPending = RegInit(false.B)
  val mvoverRobId   = Reg(chiselTypeOf(operation.rob_id))

  io.mvover.command.valid             := io.issue.valid && isMvover && !mvoverPending
  io.mvover.command.bits.sourceThread := operation.cmd.cmd.rs1Data(7, 0)
  io.mvover.command.bits.targetThread := operation.cmd.cmd.rs1Data(15, 8)
  io.mvover.command.bits.sourceBank   := operation.cmd.cmd.rs1Data(25, 16)
  io.mvover.command.bits.targetBank   := operation.cmd.cmd.rs1Data(35, 26)
  io.mvover.command.bits.sourceAddr   := operation.cmd.cmd.rs2Data(15, 0)
  io.mvover.command.bits.targetAddr   := operation.cmd.cmd.rs2Data(31, 16)
  io.mvover.command.bits.rows         := operation.cmd.cmd.rs2Data(47, 32) +& 1.U
  when(io.mvover.command.fire) {
    mvoverPending := true.B
    mvoverRobId   := operation.rob_id
  }

  decoder.io.request.valid := io.issue.valid && !isMvover && !mvoverPending
  decoder.io.request.bits  := operation.cmd
  io.issue.ready           := !mvoverPending && Mux(isMvover, io.mvover.command.ready, decoder.io.request.ready)

  val decoded = decoder.io.command
  val control =
    decoded.bits.is_config && decoded.bits.bank_id < memParam.virtualBankCount.U
  configer.io.request.valid        := decoded.valid && control
  configer.io.request.bits.cmd     := decoded.bits
  configer.io.request.bits.rob_id  := operation.rob_id
  configer.io.request.bits.context := io.issue.bits.context
  configer.io.request.bits.instId  := operation.instId
  io.request.valid                 := decoded.valid && !control
  io.request.bits.cmd              := decoded.bits
  io.request.bits.rob_id           := operation.rob_id
  io.request.bits.context          := io.issue.bits.context
  io.request.bits.instId           := operation.instId
  decoded.ready                    := Mux(control, configer.io.request.ready, io.request.ready)

  configer.io.clearBusy := io.clearBusy
  io.config <> configer.io.config

  val completion = Instantiate(new Arbiter(new Response(log2Up(frontendParam.rob_entries)), 2))
  completion.io.in(0) <> io.response
  completion.io.in(1) <> configer.io.response

  val moved = mvoverPending && io.mvover.completion.valid
  io.complete.valid          := moved || completion.io.out.valid
  io.complete.bits.fault     := Mux(moved, 0.U.asTypeOf(new DmaStatus), completion.io.out.bits.fault)
  io.complete.bits.rob_id    := Mux(moved, mvoverRobId, completion.io.out.bits.rob_id)
  completion.io.out.ready    := io.complete.ready && !moved
  io.mvover.completion.ready := io.complete.ready && mvoverPending
  when(io.mvover.completion.fire) {
    assert(!io.mvover.completion.bits, "Bank movement violated its admitted layout")
    mvoverPending := false.B
  }
}
