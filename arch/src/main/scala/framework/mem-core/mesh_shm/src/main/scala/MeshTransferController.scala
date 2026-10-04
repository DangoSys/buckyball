package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** A row-range command uses routed private-Core endpoints, without a staging Bank. */
@instantiable
class MeshTransferController(p: MeshSharedMemParams) extends Module {

  @public val io = IO(new Bundle {
    val command    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val completion = Decoupled(new MeshTransferCompletion(p))

    val mesh = new Bundle {
      val request  = Decoupled(new MeshPacket(p))
      val response = Flipped(Decoupled(new MeshPacket(p)))
    }

  })

  val Seq(idle, sourceRequest, sourceResponse, targetRequest, targetResponse, complete) = Enum(6)
  val state                                                                             = RegInit(idle)
  val command                                                                           = Reg(new MeshTransferCommand(p))
  val data                                                                              = Reg(UInt(p.dataBits.W))
  val remaining                                                                         = Reg(UInt(17.W))
  val failed                                                                            = RegInit(false.B)
  val endpoints                                                                         = VecInit(p.cores.map(core => core.bankIds.nonEmpty.B))
  val coreRows                                                                          = VecInit(p.coreLocations.map(_._1.U(p.rowBits.W)))
  val coreCols                                                                          = VecInit(p.coreLocations.map(_._2.U(p.colBits.W)))
  io.command.ready         := state === idle
  io.completion.valid      := state === complete
  io.completion.bits.tag   := command.tag
  io.completion.bits.error := failed
  when(io.command.fire) {
    command   := io.command.bits
    remaining := io.command.bits.rows
    failed    := false.B
    val input  = io.command.bits
    val source = input.sourceCore(p.coreBits - 1, 0)
    val target = input.targetCore(p.coreBits - 1, 0)
    when(input.sourceCore >= p.cores.size.U || input.targetCore >= p.cores.size.U ||
      !endpoints(source) || !endpoints(target) || input.rows === 0.U ||
      (input.sourceAddr +& input.rows) > p.entriesPerBank.U ||
      (input.targetAddr +& input.rows) > p.entriesPerBank.U ||
      (input.sourceCore === input.targetCore && input.sourceBank === input.targetBank &&
        input.sourceAddr =/= input.targetAddr &&
        input.sourceAddr < (input.targetAddr +& input.rows) &&
        input.targetAddr < (input.sourceAddr +& input.rows))) {
      failed := true.B
      state  := complete
    }.otherwise(state := sourceRequest)
  }
  val write = state === targetRequest
  val core    = Mux(write, command.targetCore, command.sourceCore)
  val slot    = core(p.coreBits - 1, 0)
  val bank    = Mux(write, command.targetBank, command.sourceBank)
  val address = Mux(write, command.targetAddr, command.sourceAddr)
  io.mesh.request.valid            := state === sourceRequest || state === targetRequest
  io.mesh.request.bits             := 0.U.asTypeOf(new MeshPacket(p))
  io.mesh.request.bits.tdest       := Cat(coreRows(slot), coreCols(slot))
  io.mesh.request.bits.tuser       := Cat(
    core,
    bank,
    true.B,
    0.U(p.rowBits.W),
    0.U(p.colBits.W),
    p.totalChannels.U(p.channelBits.W),
    address(p.addressBits - 1, 0),
    write,
    false.B
  )
  io.mesh.request.bits.tdata       := data
  io.mesh.request.bits.tkeep       := Fill(p.maskBits, 1.U(1.W))
  io.mesh.request.bits.tlast       := true.B
  io.mesh.request.bits.tid         := command.tag
  io.mesh.response.ready           := state === sourceResponse || state === targetResponse
  when(io.mesh.request.fire)(state := Mux(write, targetResponse, sourceResponse))
  when(io.mesh.response.fire) {
    assert(io.mesh.response.bits.tag === command.tag)
    assert(io.mesh.response.bits.write === (state === targetResponse))
    when(io.mesh.response.bits.error) { failed := true.B; state := complete }
      .elsewhen(state === sourceResponse) { data := io.mesh.response.bits.data; state := targetRequest }
      .elsewhen(remaining === 1.U)(state := complete)
      .otherwise {
        remaining          := remaining - 1.U
        command.sourceAddr := command.sourceAddr + 1.U
        command.targetAddr := command.targetAddr + 1.U
        state              := sourceRequest
      }
  }
  when(io.completion.fire)(state   := idle)
}
