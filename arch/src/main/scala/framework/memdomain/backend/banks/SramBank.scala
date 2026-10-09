package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * SramBank: Pure SRAM bank
 * Simple read/write memory without any accumulation logic
 * Each bank is a single-port SRAM. `clear` zeroes every row locally, one row per cycle;
 * the ports refuse requests while `clearing`, so banks clear in parallel.
 */
@instantiable
class SramBank(val b: GlobalConfig) extends Module {
  require(b.memDomain.bankEntries >= 2 && b.memDomain.bankEntries <= 65536)
  val indexBits = log2Ceil(b.memDomain.bankEntries)
  val mask_len  = b.memDomain.bankMaskLen
  val mask_elem = UInt((b.memDomain.bankWidth / mask_len).W)

  @public
  val io = IO(new Bundle {
    val sramRead  = new SramReadIO(b)
    val sramWrite = new SramWriteIO(b)
    val clear     = Input(Bool())
    val clearing  = Output(Bool())
  })

  val mem = SyncReadMem(b.memDomain.bankEntries, Vec(mask_len, mask_elem))

  // -----------------------------------------------------------------------------
  // Local clear
  // -----------------------------------------------------------------------------
  val clearing = RegInit(false.B)
  val clearRow = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  io.clearing := clearing
  when(io.clear) {
    clearing := true.B
    clearRow := 0.U
  }.elsewhen(clearing) {
    clearRow                                                    := clearRow + 1.U
    when(clearRow === (b.memDomain.bankEntries - 1).U)(clearing := false.B)
  }

  val readPending = RegNext(io.sramRead.req.fire, false.B)
  val readHeld    = RegInit(false.B)
  val readData    = Reg(UInt(b.memDomain.bankWidth.W))
  val writeValid  = RegInit(false.B)

  io.sramRead.resp.valid    := readPending || readHeld
  io.sramWrite.resp.valid   := writeValid
  io.sramWrite.resp.bits.ok := true.B
  io.sramWrite.req.ready    := !io.clear && !clearing && (!writeValid || io.sramWrite.resp.ready)
  io.sramRead.req.ready     := !io.clear && !clearing && !io.sramWrite.req.fire &&
    (!io.sramRead.resp.valid || io.sramRead.resp.ready)

  val ren = io.sramRead.req.fire
  val wen = clearing || io.sramWrite.req.fire

  when(io.sramRead.req.fire) {
    assert(io.sramRead.req.bits.addr < b.memDomain.bankEntries.U, "SRAM read exceeds bank depth")
  }
  when(io.sramWrite.req.fire) {
    assert(io.sramWrite.req.bits.addr < b.memDomain.bankEntries.U, "SRAM write exceeds bank depth")
  }

  val rdata = mem.readWrite(
    Mux(
      clearing,
      clearRow,
      Mux(wen, io.sramWrite.req.bits.addr(indexBits - 1, 0), io.sramRead.req.bits.addr(indexBits - 1, 0))
    ),
    Mux(clearing, 0.U.asTypeOf(Vec(mask_len, mask_elem)), io.sramWrite.req.bits.data.asTypeOf(Vec(mask_len, mask_elem))),
    Mux(clearing, VecInit(Seq.fill(mask_len)(true.B)), io.sramWrite.req.bits.mask),
    ren || wen,
    wen
  )

  io.sramRead.resp.bits.data := Mux(readHeld, readData, rdata.asUInt)
  when(readPending && !io.sramRead.resp.ready) {
    readHeld := true.B
    readData := rdata.asUInt
  }.elsewhen(io.sramRead.resp.fire) {
    readHeld := false.B
  }
  when(io.sramWrite.req.fire) {
    writeValid := true.B
  }.elsewhen(io.sramWrite.resp.fire) {
    writeValid := false.B
  }
}
