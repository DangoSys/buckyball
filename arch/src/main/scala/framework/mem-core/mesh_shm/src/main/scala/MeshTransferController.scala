package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._

/** One transfer at a time; the staged word stays reserved until completion. */
class MeshTransferController(p: MeshSharedMemParams) extends Module {

  val io = IO(new Bundle {
    val command    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val completion = Decoupled(new MeshTransferCompletion(p))
    val localBanks = Vec(p.cores.size, new MeshLocalBankPort(p.global, p.addressBits, p.localBankBits, p.tagBits))

    val mesh = new Bundle {
      val request  = Decoupled(new MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits))
      val response = Flipped(Decoupled(new MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits)))
    }

  })

  val Seq(
    sIdle,
    sSourceRequest,
    sSourceResponse,
    sStageWriteRequest,
    sStageWriteResponse,
    sStageReadRequest,
    sStageReadResponse,
    sTargetRequest,
    sTargetResponse,
    sComplete
  ) = Enum(10)

  val state      = RegInit(sIdle)
  val command    = Reg(new MeshTransferCommand(p))
  val sourceData = Reg(UInt(p.dataBits.W))
  val stagedData = Reg(UInt(p.dataBits.W))
  val stageValid = RegInit(false.B)
  val failed     = RegInit(false.B)

  io.command.ready         := state === sIdle
  io.completion.valid      := state === sComplete
  io.completion.bits.tag   := command.tag
  io.completion.bits.error := failed

  when(io.command.fire) {
    command    := io.command.bits
    stageValid := false.B
    failed     := false.B
    when(io.command.bits.sourceCore >= p.cores.size.U ||
      io.command.bits.targetCore >= p.cores.size.U) {
      failed := true.B
      state  := sComplete
    }.otherwise {
      state := sSourceRequest
    }
  }

  for ((port, index) <- io.localBanks.zipWithIndex) {
    val isSource = command.sourceCore === index.U
    val isTarget = command.targetCore === index.U

    port.request.valid      :=
      (state === sSourceRequest && isSource) || (state === sTargetRequest && isTarget)
    port.request.bits       := 0.U.asTypeOf(port.request.bits)
    port.request.bits.tdest := Mux(state === sSourceRequest, command.sourceBank, command.targetBank)
    port.request.bits.addr  := Mux(state === sSourceRequest, command.sourceAddr, command.targetAddr)
    port.request.bits.tuser := Mux(state === sSourceRequest, MeshEvent.ReadRequest, MeshEvent.WriteRequest)
    port.request.bits.tdata := Mux(state === sTargetRequest, stagedData, 0.U)
    port.request.bits.tkeep := Mux(state === sTargetRequest, Fill(p.maskBits, 1.U(1.W)), 0.U)
    port.request.bits.tlast := true.B
    port.request.bits.tid   := command.tag
    port.response.ready     :=
      (state === sSourceResponse && isSource) || (state === sTargetResponse && isTarget)

    when(state === sSourceRequest && isSource && port.request.fire) {
      state := sSourceResponse
    }
    when(state === sSourceResponse && isSource && port.response.fire) {
      assert(port.response.bits.tlast)
      assert(port.response.bits.tid === command.tag)
      assert(port.response.bits.tuser(1, 0) === MeshEvent.ReadResponse(1, 0))
      when(port.response.bits.tuser(2)) {
        failed := true.B
        state  := sComplete
      }.otherwise {
        sourceData := port.response.bits.tdata
        state      := sStageWriteRequest
      }
    }
    when(state === sTargetRequest && isTarget && port.request.fire) {
      state := sTargetResponse
    }
    when(state === sTargetResponse && isTarget && port.response.fire) {
      assert(port.response.bits.tlast)
      assert(port.response.bits.tid === command.tag)
      assert(port.response.bits.tuser(1, 0) === MeshEvent.WriteResponse(1, 0))
      when(port.response.bits.tuser(2)) {
        failed := true.B
      }
      state := sComplete
    }
  }

  io.mesh.request.valid      := state === sStageWriteRequest || state === sStageReadRequest
  io.mesh.request.bits       := 0.U.asTypeOf(io.mesh.request.bits)
  io.mesh.request.bits.tdest := p.stagingBank.U
  io.mesh.request.bits.addr  := p.stagingAddress.U
  io.mesh.request.bits.tuser := Mux(state === sStageWriteRequest, MeshEvent.WriteRequest, MeshEvent.ReadRequest)
  io.mesh.request.bits.tdata := Mux(state === sStageWriteRequest, sourceData, 0.U)
  io.mesh.request.bits.tkeep := Mux(state === sStageWriteRequest, Fill(p.maskBits, 1.U(1.W)), 0.U)
  io.mesh.request.bits.tlast := true.B
  io.mesh.request.bits.tid   := command.tag
  io.mesh.response.ready     := state === sStageWriteResponse || state === sStageReadResponse

  when(io.mesh.request.fire) {
    when(state === sStageWriteRequest) {
      state := sStageWriteResponse
    }.otherwise {
      assert(state === sStageReadRequest && stageValid)
      state := sStageReadResponse
    }
  }
  when(io.mesh.response.fire) {
    assert(io.mesh.response.bits.tlast)
    assert(io.mesh.response.bits.tid === command.tag)
    assert(io.mesh.response.bits.tuser(1, 0) === Mux(
      state === sStageWriteResponse,
      MeshEvent.WriteResponse(1, 0),
      MeshEvent.ReadResponse(1, 0)
    ))
    when(io.mesh.response.bits.tuser(2)) {
      failed := true.B
      state  := sComplete
    }.elsewhen(state === sStageWriteResponse) {
      stageValid := true.B
      state      := sStageReadRequest
    }.otherwise {
      assert(state === sStageReadResponse && stageValid)
      stagedData := io.mesh.response.bits.tdata
      state      := sTargetRequest
    }
  }

  when(io.completion.fire) {
    stageValid := false.B
    state      := sIdle
  }
}
