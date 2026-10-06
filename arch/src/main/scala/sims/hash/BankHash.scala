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
    val write      = Flipped(Decoupled(new BankHashWrite(b)))
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

  val writeValid      = RegInit(false.B)
  val writeBits       = Reg(new BankHashWrite(b))
  val writeGeneration = Reg(UInt(generation.getWidth.W))
  val commit          = writeValid && writeGeneration === generation && !io.bind
  io.write.ready := !writeValid && !io.bind
  val oldRow     = Wire(UInt(128.W))
  val oldBytes   = oldRow.asTypeOf(Vec(16, UInt(8.W)))
  val writeBytes = writeBits.data.asTypeOf(Vec(16, UInt(8.W)))
  val newBytes   = VecInit((0 until 16).map(i => Mux(writeBits.mask(i), writeBytes(i), oldBytes(i))))
  val newRow     = newBytes.asUInt
  val nextHash   = statusHash - rowHash(writeBits.addr, oldRow) + rowHash(writeBits.addr, newRow)

  val readData =
    shadow.readWrite(Mux(commit, writeBits.addr, io.write.bits.addr), newRow, io.write.fire || commit, commit)
  oldRow := Mux(validRows(writeBits.addr), readData, 0.U)

  when(io.write.fire) {
    writeValid      := true.B
    writeBits       := io.write.bits
    writeGeneration := generation
  }

  io.statusHash := statusHash

  when(io.bind) {
    generation          := generation + 1.U
    statusHash          := 0.U
    writeValid          := false.B
    validRows.foreach(_ := false.B)
  }.elsewhen(commit) {
    validRows(writeBits.addr) := true.B
    statusHash                := nextHash
    writeValid                := false.B
  }
}
