package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._

/** One transfer at a time; the staged word stays reserved until completion. */
class MeshTransferController(p: MeshSharedMemParams) extends Module {

  val io = IO(new Bundle {
    val command    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val completion = Decoupled(new MeshTransferCompletion(p))
    val localBanks = Vec(p.cores.size, new MeshLocalBankPort(p))

    val mesh = new Bundle {
      val request  = Decoupled(new MeshClientRequest(p))
      val response = Flipped(Decoupled(new MeshClientResponse(p)))
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
    port.request.bits.addr  := Mux(state === sSourceRequest, command.sourceAddr, command.targetAddr)
    port.request.bits.write := state === sTargetRequest
    port.request.bits.data  := stagedData
    port.request.bits.mask  := Fill(p.maskBits, 1.U(1.W))
    port.request.bits.tag   := command.tag
    port.response.ready     :=
      (state === sSourceResponse && isSource) || (state === sTargetResponse && isTarget)

    when(state === sSourceRequest && isSource && port.request.fire) {
      state := sSourceResponse
    }
    when(state === sSourceResponse && isSource && port.response.fire) {
      assert(port.response.bits.tag === command.tag)
      when(port.response.bits.error) {
        failed := true.B
        state  := sComplete
      }.otherwise {
        sourceData := port.response.bits.data
        state      := sStageWriteRequest
      }
    }
    when(state === sTargetRequest && isTarget && port.request.fire) {
      state := sTargetResponse
    }
    when(state === sTargetResponse && isTarget && port.response.fire) {
      assert(port.response.bits.tag === command.tag)
      when(port.response.bits.error) {
        failed := true.B
      }
      state := sComplete
    }
  }

  io.mesh.request.valid      := state === sStageWriteRequest || state === sStageReadRequest
  io.mesh.request.bits       := 0.U.asTypeOf(new MeshClientRequest(p))
  io.mesh.request.bits.bank  := p.stagingBank.U
  io.mesh.request.bits.addr  := p.stagingAddress.U
  io.mesh.request.bits.write := state === sStageWriteRequest
  io.mesh.request.bits.data  := sourceData
  io.mesh.request.bits.mask  := Fill(p.maskBits, 1.U(1.W))
  io.mesh.request.bits.tag   := command.tag
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
    assert(io.mesh.response.bits.tag === command.tag)
    when(io.mesh.response.bits.error) {
      failed := true.B
      state  := sComplete
    }.elsewhen(state === sStageWriteResponse) {
      stageValid := true.B
      state      := sStageReadRequest
    }.otherwise {
      assert(state === sStageReadResponse && stageValid)
      stagedData := io.mesh.response.bits.data
      state      := sTargetRequest
    }
  }

  when(io.completion.fire) {
    stageValid := false.B
    state      := sIdle
  }
}
