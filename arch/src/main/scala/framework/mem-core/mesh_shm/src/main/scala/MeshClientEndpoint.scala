package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Converts one client operation class into packets injected at one Mesh node. */
@instantiable
class MeshClientEndpoint(
  p:             MeshSharedMemParams,
  row:           Int,
  col:           Int,
  channel:       Int,
  acceptedWrite: Option[Boolean])
    extends Module {
  require(row >= 0 && row < p.rows && col >= 0 && col < p.cols)
  require(channel >= 0 && channel < p.networkClientChannels)

  @public
  val io = IO(new Bundle {
    val client   = new MeshChannel(p)
    val inFlight = Output(UInt(log2Ceil(p.maxInFlight + 1).W))
    val tags     = Output(UInt((1 << p.tagBits).W))

    val mesh = new Bundle {
      val request  = Decoupled(new MeshPacket(p))
      val response = Flipped(Decoupled(new MeshPacket(p)))
    }

  })

  val inFlight     = RegInit(0.U(log2Ceil(p.maxInFlight + 1).W))
  val tags         = RegInit(0.U((1 << p.tagBits).W))
  val errorPending = RegInit(false.B)
  val errorTag     = Reg(UInt(p.tagBits.W))
  val errorWrite   = Reg(Bool())
  val request      = io.client.request
  val response     = io.client.response
  val available    = inFlight < p.maxInFlight.U && !tags(request.bits.tag)
  val bankValid    = request.bits.bank < p.visibleBankCount.U
  val bankRows     = VecInit((0 until p.bankCount).map(i => (i / p.cols).U(p.rowBits.W)))
  val bankCols     = VecInit((0 until p.bankCount).map(i => (i % p.cols).U(p.colBits.W)))
  io.inFlight := inFlight
  io.tags     := tags

  acceptedWrite.foreach { write =>
    when(request.valid) {
      assert(request.bits.write === write.B, "Mesh client operation used the wrong directional endpoint")
    }
  }

  io.mesh.request.valid      := request.valid && available && !errorPending && bankValid
  io.mesh.request.bits       := 0.U.asTypeOf(new MeshPacket(p))
  io.mesh.request.bits.tdest := Cat(bankRows(request.bits.bank), bankCols(request.bits.bank))
  io.mesh.request.bits.tdata := request.bits.data
  io.mesh.request.bits.tkeep := request.bits.mask
  io.mesh.request.bits.tid   := request.bits.tag
  io.mesh.request.bits.tlast := true.B
  io.mesh.request.bits.tuser := Cat(
    row.U(p.rowBits.W),
    col.U(p.colBits.W),
    channel.U(p.channelBits.W),
    request.bits.addr,
    request.bits.write,
    false.B
  )
  request.ready              := available && !errorPending && Mux(bankValid, io.mesh.request.ready, true.B)
  when(request.fire && !bankValid) {
    errorPending := true.B
    errorTag     := request.bits.tag
    errorWrite   := request.bits.write
  }

  val responseValid = RegInit(false.B)
  val responseBits  = Reg(new MeshClientResponse(p))
  val responseNext  = Wire(Decoupled(new MeshClientResponse(p)))
  response.valid          := responseValid
  response.bits           := responseBits
  responseNext.ready      := !responseValid || response.ready
  when(responseNext.ready) {
    responseValid                         := responseNext.valid
    when(responseNext.valid)(responseBits := responseNext.bits)
  }
  responseNext.valid      := errorPending || io.mesh.response.valid
  responseNext.bits       := 0.U.asTypeOf(responseNext.bits)
  responseNext.bits.data  := Mux(errorPending, 0.U, io.mesh.response.bits.data)
  responseNext.bits.tag   := Mux(errorPending, errorTag, io.mesh.response.bits.tag)
  responseNext.bits.tkeep := Fill(p.maskBits, 1.U(1.W))
  responseNext.bits.tlast := true.B
  responseNext.bits.tuser := Cat(
    Mux(errorPending, errorWrite, io.mesh.response.bits.write),
    errorPending || io.mesh.response.bits.error
  )
  io.mesh.response.ready  := responseNext.ready && !errorPending

  when(io.mesh.response.valid) {
    assert(io.mesh.response.bits.channel === channel.U)
    assert(inFlight =/= 0.U && tags(io.mesh.response.bits.tag))
  }
  when(response.fire) {
    assert(inFlight =/= 0.U && tags(response.bits.tag))
  }
  when(responseNext.fire && errorPending)(errorPending := false.B)
  val issued    = Mux(request.fire, 1.U((1 << p.tagBits).W) << request.bits.tag, 0.U)
  val completed = Mux(response.fire, 1.U((1 << p.tagBits).W) << response.bits.tag, 0.U)
  tags := (tags | issued) & ~completed
  when(request.fire =/= response.fire) {
    inFlight := Mux(request.fire, inFlight + 1.U, inFlight - 1.U)
  }
}
