package framework.system.tile.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.shared.SharedPhysicalPort
import memcore.bus.axi4

/**
 * AXI access to the existing Tile shared SRAM. No storage is instantiated here.
 * Read and write bursts each have one outstanding transaction and independent ports.
 * W acceptance is not completion: B follows the actual final shared-bank write acknowledgement.
 */
@instantiable
class SharedAxiEndpoint(p: T2TParams) extends Module {

  @public val io = IO(new Bundle {
    val busy  = Output(Bool())
    val mem   = Flipped(new axi4.Port(p.axi))
    val read  = new SharedPhysicalPort(128)
    val write = new SharedPhysicalPort(128)
  })

  val rIdle :: rIssue :: rWait :: rReply :: Nil          = Enum(4)
  val wIdle :: wData :: wIssue :: wWait :: wReply :: Nil = Enum(5)
  val readState                                          = RegInit(rIdle)
  val writeState                                         = RegInit(wIdle)
  val readAddress                                        = Reg(UInt(p.sharedAddressBits.W))
  val writeAddress                                       = Reg(UInt(p.sharedAddressBits.W))
  val readId                                             = Reg(UInt(p.axi.idBits.W))
  val writeId                                            = Reg(UInt(p.axi.idBits.W))
  val readRemaining                                      = Reg(UInt(9.W))
  val writeRemaining                                     = Reg(UInt(9.W))
  val readData                                           = Reg(UInt(128.W))
  val writeData                                          = Reg(UInt(128.W))
  val writeMask                                          = Reg(UInt(16.W))
  io.busy := readState =/= rIdle || writeState =/= wIdle
  val live = !reset.asBool

  def check(address: axi4.Address): Unit = {
    val bytes = (address.len +& 1.U) << 4
    assert(
      address.size === 4.U && address.burst === 1.U && !address.lock && address.addr(3, 0) === 0.U,
      "T2T shared SRAM requires unlocked aligned 128-bit INCR bursts"
    )
    assert(address.addr +& bytes <= p.sharedBytes.U, "T2T AXI address exceeds Tile shared SRAM")
    assert(address.addr.pad(12)(11, 0) +& bytes <= 4096.U, "T2T AXI burst crosses 4KiB")
    assert((address.addr & (p.bankBytes - 1).U) +& bytes <= p.bankBytes.U, "T2T AXI burst crosses a shared bank")
  }

  io.mem.ar.ready                      := readState === rIdle && live
  when(io.mem.ar.fire) {
    check(io.mem.ar.bits)
    readAddress   := io.mem.ar.bits.addr; readId          := io.mem.ar.bits.id
    readRemaining := io.mem.ar.bits.len +& 1.U; readState := rIssue
  }
  io.read.request.valid                := readState === rIssue && live
  io.read.request.bits.address         := readAddress
  io.read.request.bits.write           := false.B
  io.read.request.bits.data            := 0.U
  io.read.request.bits.mask            := 0.U
  when(io.read.request.fire)(readState := rWait)
  io.read.response.ready               := readState === rWait && live
  when(io.read.response.fire) {
    assert(!io.read.response.bits.error, "T2T shared SRAM read failed")
    when(!io.read.response.bits.error) { readData := io.read.response.bits.data; readState := rReply }
  }
  io.mem.r.valid                       := readState === rReply && live
  io.mem.r.bits.id                     := readId
  io.mem.r.bits.data                   := readData
  io.mem.r.bits.resp                   := 0.U
  io.mem.r.bits.last                   := readRemaining === 1.U
  when(io.mem.r.fire) {
    when(readRemaining === 1.U)(readState := rIdle).otherwise {
      readAddress := readAddress + 16.U; readRemaining := readRemaining - 1.U; readState := rIssue
    }
  }

  io.mem.aw.ready                        := writeState === wIdle && live
  when(io.mem.aw.fire) {
    check(io.mem.aw.bits)
    writeAddress   := io.mem.aw.bits.addr; writeId          := io.mem.aw.bits.id
    writeRemaining := io.mem.aw.bits.len +& 1.U; writeState := wData
  }
  io.mem.w.ready                         := writeState === wData && live
  when(io.mem.w.fire) {
    assert(io.mem.w.bits.last === (writeRemaining === 1.U), "T2T AXI WLAST does not match AWLEN")
    writeData := io.mem.w.bits.data; writeMask := io.mem.w.bits.strb; writeState := wIssue
  }
  io.write.request.valid                 := writeState === wIssue && live
  io.write.request.bits.address          := writeAddress
  io.write.request.bits.write            := true.B
  io.write.request.bits.data             := writeData
  io.write.request.bits.mask             := writeMask
  when(io.write.request.fire)(writeState := wWait)
  io.write.response.ready                := writeState === wWait && live
  when(io.write.response.fire) {
    assert(!io.write.response.bits.error, "T2T shared SRAM write failed")
    when(!io.write.response.bits.error) {
      when(writeRemaining === 1.U)(writeState := wReply).otherwise {
        writeAddress := writeAddress + 16.U; writeRemaining := writeRemaining - 1.U; writeState := wData
      }
    }
  }
  io.mem.b.valid                         := writeState === wReply && live
  io.mem.b.bits.id                       := writeId
  io.mem.b.bits.resp                     := 0.U
  when(io.mem.b.fire)(writeState         := wIdle)
}
