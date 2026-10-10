package framework.system.tile.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.memdomain.backend.shared.SharedPhysicalPort
import memcore.bus.axi4
import memcore.memory.queue.Queue

case class T2TParams(
  sharedBytes: BigInt,
  bankBytes:   Int,
  tiles:       Int,
  axi:         axi4.Params) {
  require(sharedBytes > 0 && bankBytes >= 16 && isPow2(bankBytes) && sharedBytes % bankBytes == 0)
  require(tiles > 0 && axi.dataBits == 128)
  val sharedAddressBits: Int = math.max(1, log2Ceil(sharedBytes))
  val tileBits:          Int = math.max(1, log2Ceil(tiles))
  require(sharedAddressBits + tileBits <= axi.addressBits && sharedAddressBits <= 32)
}

class T2TCommand(p: T2TParams) extends Bundle {
  val sourceByteAddress = UInt(64.W)
  val targetTile        = UInt(p.tileBits.W)
  val targetByteAddress = UInt(64.W)
  val bytes             = UInt(32.W)
}

/**
 * Push shared SRAM bytes to another Tile. Starts are 16-byte aligned; the final beat may be partial.
 * One command is active. Bank mappings remain owned by the caller through completion.
 * Reset is coordinated with the fabric and SRAM ports, not an independent transaction cancellation.
 */
@instantiable
class T2TTransfer(p: T2TParams) extends Module {

  @public val io = IO(new Bundle {
    val command    = Flipped(Decoupled(new T2TCommand(p)))
    val completion = Decoupled(Bool()) // false = no error, after the final destination write response
    val source     = new SharedPhysicalPort(128)
    val mem        = new axi4.Port(p.axi)
  })

  val idle :: plan :: stream :: reply :: complete :: Nil = Enum(5)
  val state                                              = RegInit(idle)
  val source                                             = Reg(UInt(p.sharedAddressBits.W))
  val target                                             = Reg(UInt(p.sharedAddressBits.W))
  val tile                                               = Reg(UInt(p.tileBits.W))
  val remaining                                          = Reg(UInt((p.sharedAddressBits + 1).W))
  val burstBytes                                         = Reg(UInt(13.W))
  val beats                                              = Reg(UInt(9.W))
  val issued                                             = RegInit(0.U(9.W))
  val sent                                               = RegInit(0.U(9.W))
  val pending                                            = RegInit(false.B)
  val awSent                                             = RegInit(false.B)
  val data                                               = Instantiate(new Queue(UInt(128.W), 2))
  val live                                               = !reset.asBool
  io.command.ready               := state === idle && live
  io.completion.valid            := state === complete && live
  io.completion.bits             := false.B
  when(io.completion.fire)(state := idle)
  when(io.command.fire) {
    val cmd = io.command.bits
    assert(cmd.bytes =/= 0.U && cmd.sourceByteAddress(3, 0) === 0.U && cmd.targetByteAddress(3, 0) === 0.U)
    assert(cmd.targetTile < p.tiles.U, "T2T target must be a Tile system ID")
    assert(
      cmd.sourceByteAddress +& cmd.bytes <= p.sharedBytes.U && cmd.targetByteAddress +& cmd.bytes <= p.sharedBytes.U,
      "T2T command exceeds Tile shared SRAM"
    )
    source := cmd.sourceByteAddress; target := cmd.targetByteAddress
    tile   := cmd.targetTile; remaining     := cmd.bytes; state := plan
  }
  def minimum(a: UInt, b: UInt): UInt = Mux(a < b, a, b)
  val sourcePage = 4096.U(13.W) - source.pad(12)(11, 0)
  val targetPage = 4096.U(13.W) - target.pad(12)(11, 0)
  val sourceBank = p.bankBytes.U - (source & (p.bankBytes - 1).U)
  val targetBank = p.bankBytes.U - (target & (p.bankBytes - 1).U)
  val nextBytes  = Seq(remaining, sourcePage, targetPage, sourceBank, targetBank, 4096.U).reduce(minimum)
  when(state === plan) {
    burstBytes := nextBytes; beats := (nextBytes +& 15.U) >> 4
    issued     := 0.U; sent        := 0.U; pending := false.B; awSent := false.B; state := stream
    assert(data.io.count === 0.U && nextBytes =/= 0.U)
  }
  io.mem.aw.valid := state === stream && !awSent && live
  io.mem.aw.bits                        := 0.U.asTypeOf(new axi4.Address(p.axi))
  io.mem.aw.bits.addr                   := Cat(tile, target(p.sharedAddressBits - 1, 0))
  io.mem.aw.bits.len                    := beats - 1.U
  io.mem.aw.bits.size                   := 4.U
  io.mem.aw.bits.burst                  := 1.U
  when(io.mem.aw.fire)(awSent           := true.B)
  io.source.request.valid               := state === stream && issued < beats && !pending && data.io.count < 2.U && live
  io.source.request.bits.address        := source + (issued << 4)
  io.source.request.bits.write          := false.B
  io.source.request.bits.data           := 0.U
  io.source.request.bits.mask           := 0.U
  when(io.source.request.fire) { pending := true.B; issued := issued + 1.U }
  data.io.enq.valid                     := state === stream && pending && io.source.response.valid && !io.source.response.bits.error && live
  data.io.enq.bits                      := io.source.response.bits.data
  io.source.response.ready              := state === stream && pending && data.io.enq.ready && live
  when(io.source.response.valid && live) {
    assert(pending && !io.source.response.bits.error, "T2T shared source read failed")
  }
  when(io.source.response.fire)(pending := false.B)
  val last = sent +& 1.U === beats
  val tail = burstBytes(3, 0)
  io.mem.w.valid     := state === stream && awSent && data.io.deq.valid && live
  io.mem.w.bits.data := data.io.deq.bits
  io.mem.w.bits.strb := Mux(last && tail =/= 0.U, (1.U(17.W) << tail) - 1.U, "hffff".U)
  io.mem.w.bits.last := last
  data.io.deq.ready  := state === stream && awSent && io.mem.w.ready && live
  when(io.mem.w.fire) { sent := sent + 1.U; when(last)(state := reply) }
  io.mem.b.ready     := state === reply && live
  when(io.mem.b.fire) {
    assert(io.mem.b.bits.id === 0.U && io.mem.b.bits.resp === 0.U, "T2T destination write failed")
    assert(!pending && data.io.count === 0.U && issued === beats && sent === beats)
    when(io.mem.b.bits.id === 0.U && io.mem.b.bits.resp === 0.U) {
      when(remaining === burstBytes)(state := complete).otherwise {
        source    := source + burstBytes; target   := target + burstBytes
        remaining := remaining - burstBytes; state := plan
      }
    }
  }
  io.mem.ar.valid    := false.B
  io.mem.ar.bits     := 0.U.asTypeOf(new axi4.Address(p.axi))
  io.mem.r.ready     := false.B
  when(live)(assert(!io.mem.r.valid, "T2T push engine received an unsolicited AXI read response"))
}
