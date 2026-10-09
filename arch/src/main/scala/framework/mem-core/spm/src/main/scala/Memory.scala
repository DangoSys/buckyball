package memcore.memory.spm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.bank.Bank

/**
 * One outstanding transaction on one physical read/write port. Writes acknowledge after
 * SRAM acceptance; reads return after the synchronous read. Responses hold until accepted.
 * Reset cancels protocol state; SRAM contents, including already accepted writes, remain.
 */
@instantiable
class Memory(p: Params) extends Module {

  @public val io = IO(new Bundle {
    val port     = Flipped(new Port(p))
    val readOnly = Input(Bool())
    val busy     = Output(Bool())
  })

  val bank                              = Instantiate(new Bank(p.bank))
  val idle :: memory :: rejected :: Nil = Enum(3)
  val state                             = RegInit(idle)
  val offset                            = Reg(UInt(p.laneBits.W))
  val savedMask                         = Reg(UInt(p.beatBytes.W))
  val request                           = io.port.request.bits
  val count                             = 1.U(64.W) << request.size
  val end                               = request.address +& count

  val widthMask = MuxLookup(request.size, 0.U(p.beatBytes.W))(
    (0 to p.laneBits).map(size => size.U -> ((BigInt(1) << (1 << size)) - 1).U(p.beatBytes.W))
  )

  val legal = request.size <= p.laneBits.U && (request.address & (count - 1.U)) === 0.U &&
    request.address >= p.base.U && end <= (p.base + p.bytes).U &&
    !(request.write && (io.readOnly || (request.mask & ~widthMask) =/= 0.U))

  val lane = request.address(p.laneBits - 1, 0)
  io.port.request.ready             := state === idle && !reset.asBool && Mux(legal, bank.io.request.ready, true.B)
  io.busy                           := state =/= idle
  bank.io.request.valid             := state === idle && !reset.asBool && io.port.request.valid && legal
  bank.io.request.bits.addr         := (request.address - p.base.U) >> p.laneBits
  bank.io.request.bits.write        := request.write
  bank.io.request.bits.data         := request.data << (lane << 3)
  bank.io.request.bits.mask         := request.mask << lane
  bank.io.request.bits.tag          := 0.U
  when(io.port.request.fire) {
    offset    := lane
    savedMask := widthMask
    state     := Mux(legal, memory, rejected)
  }
  val dataMask = Cat((0 until p.beatBytes).reverse.map(i => Fill(8, savedMask(i))))
  io.port.response.valid            := !reset.asBool && (state === rejected || (state === memory && bank.io.response.valid))
  io.port.response.bits.data        := Mux(state === rejected, 0.U, (bank.io.response.bits.data >> (offset << 3)) & dataMask)
  io.port.response.bits.error       := state === rejected || bank.io.response.bits.error
  bank.io.response.ready            := state === memory && io.port.response.ready && !reset.asBool
  when(io.port.response.fire)(state := idle)
}
