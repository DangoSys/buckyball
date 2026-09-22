package sims.hash

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.top.GlobalConfig

class BankHashWrite(val b: GlobalConfig) extends Bundle {
  val addr = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val mask = Vec(b.memDomain.bankMaskLen, Bool())
  val data = UInt(b.memDomain.bankWidth.W)
}

@instantiable
class BankHashMonitor(val b: GlobalConfig) extends Module {
  require(b.memDomain.bankWidth == 128, "bank hash requires 128-bit bank rows")
  require(b.memDomain.bankMaskLen == 16, "bank hash requires byte write masks")

  @public
  val io = IO(new Bundle {
    val bind       = Input(Bool())
    val write      = Flipped(Valid(new BankHashWrite(b)))
    val statusHash = Output(UInt(32.W))
  })

  val shadow     = SyncReadMem(b.memDomain.bankEntries, UInt(128.W))
  val validRows  = RegInit(VecInit(Seq.fill(b.memDomain.bankEntries)(false.B)))
  val generation = RegInit(0.U(8.W))
  val statusHash = RegInit(0.U(32.W))

  def rotateLeft(value: UInt, amount: Int): UInt =
    Cat(value(31 - amount, 0), value(31, 32 - amount))

  def rowHash(addr: UInt, row: UInt): UInt = {
    val address = addr.pad(32)
    val mixed   = row(31, 0) ^ rotateLeft(row(63, 32), 7) ^ rotateLeft(row(95, 64), 13) ^
      rotateLeft(row(127, 96), 21) ^ rotateLeft(address, 11)
    Mux(row.orR, mixed, 0.U)
  }

  val readData        = shadow.read(io.write.bits.addr, io.write.valid)
  val writeValid      = RegNext(io.write.valid, false.B)
  val writeBits       = RegEnable(io.write.bits, io.write.valid)
  val writeGeneration = RegEnable(generation, io.write.valid)
  val lastValid       = RegInit(false.B)
  val lastAddr        = Reg(UInt(io.write.bits.addr.getWidth.W))
  val lastData        = Reg(UInt(128.W))
  val lastGeneration  = Reg(UInt(generation.getWidth.W))
  val memoryOld       = Mux(validRows(writeBits.addr), readData, 0.U)
  val bypass          = lastValid && lastGeneration === writeGeneration && lastAddr === writeBits.addr
  val oldRow          = Mux(bypass, lastData, memoryOld)
  val oldBytes        = oldRow.asTypeOf(Vec(16, UInt(8.W)))
  val writeBytes      = writeBits.data.asTypeOf(Vec(16, UInt(8.W)))
  val newBytes        = VecInit((0 until 16).map(i => Mux(writeBits.mask(i), writeBytes(i), oldBytes(i))))
  val newRow          = newBytes.asUInt
  val nextHash        = statusHash - rowHash(writeBits.addr, oldRow) + rowHash(writeBits.addr, newRow)
  val commit          = writeValid && writeGeneration === generation && !io.bind

  io.statusHash := statusHash

  when(io.bind) {
    generation          := generation + 1.U
    statusHash          := 0.U
    lastValid           := false.B
    validRows.foreach(_ := false.B)
  }.elsewhen(commit) {
    shadow.write(writeBits.addr, newRow)
    validRows(writeBits.addr) := true.B
    statusHash                := nextHash
    lastValid                 := true.B
    lastAddr                  := writeBits.addr
    lastData                  := newRow
    lastGeneration            := writeGeneration
  }
}
