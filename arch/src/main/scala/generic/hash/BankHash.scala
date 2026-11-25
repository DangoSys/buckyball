package framework.generic.hash

/** The shape of one bank: `entries` rows of `width` bits written under `maskLen` mask bits. */
case class BankGeometry(entries: Int, width: Int, maskLen: Int)

import chisel3._
import chisel3.experimental.IntParam
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._

class BankHashWrite(val b: BankGeometry) extends Bundle {
  val addr = UInt(log2Ceil(b.entries).W)
  val mask = Vec(b.maskLen, Bool())
  val data = UInt(b.width.W)
}

@instantiable
class BankHashStorage(val b: BankGeometry)
    extends BlackBox(Map("DEPTH" -> IntParam(b.entries)))
    with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock        = Input(Clock())
    val readEnable   = Input(Bool())
    val readAddress  = Input(UInt(log2Ceil(b.entries).W))
    val readData     = Output(UInt(128.W))
    val writeEnable  = Input(Bool())
    val writeAddress = Input(UInt(log2Ceil(b.entries).W))
    val writeData    = Input(UInt(128.W))
  })

  // Keep the synchronous data register inside the memory inference boundary.
  setInline(
    "BankHashStorage.sv",
    """
      |module BankHashStorage #(parameter DEPTH = 2) (
      |  input clock,
      |  input readEnable,
      |  input [$clog2(DEPTH)-1:0] readAddress,
      |  output reg [127:0] readData,
      |  input writeEnable,
      |  input [$clog2(DEPTH)-1:0] writeAddress,
      |  input [127:0] writeData
      |);
      |  (* syn_ramstyle = "ultra" *) reg [127:0] storage [0:DEPTH-1];
      |  always @(posedge clock) begin
      |    if (readEnable) readData <= storage[readAddress];
      |    if (writeEnable) storage[writeAddress] <= writeData;
      |  end
      |endmodule
      |""".stripMargin
  )
}

@instantiable
class BankHashMonitor(val b: BankGeometry) extends Module {
  require(b.width == 128, "bank hash requires 128-bit bank rows")
  require(b.maskLen == 16, "bank hash requires byte write masks")

  @public
  val io = IO(new Bundle {
    val bind       = Input(Bool())
    val write      = Flipped(Decoupled(new BankHashWrite(b)))
    val statusHash = Output(UInt(32.W))
  })

  // Pipeline the shadow read so it maps to SRAM while accepting a write every cycle.
  val shadow           = Instantiate(new BankHashStorage(b))
  val validityGroups   = (b.entries + 31) / 32
  val validGroups      = RegInit(0.U(validityGroups.W))
  val validityWords    = SyncReadMem(validityGroups, UInt(32.W))
  val groupWasValid    = Reg(Bool())
  val forwardGroup     = Reg(Bool())
  val previousValidity = Reg(UInt(32.W))
  val statusHash       = RegInit(0.U(32.W))
  val pending          = RegInit(false.B)
  val write            = Reg(new BankHashWrite(b))
  val forward          = Reg(Bool())
  val previousRow      = Reg(UInt(128.W))

  def rotateLeft(value: UInt, amount: Int): UInt =
    Cat(value(31 - amount, 0), value(31, 32 - amount))

  def rowHash(addr: UInt, row: UInt): UInt = {
    val address = addr.pad(32)
    val mixed   = row(31, 0) ^ rotateLeft(row(63, 32), 7) ^ rotateLeft(row(95, 64), 13) ^
      rotateLeft(row(127, 96), 21) ^ rotateLeft(address, 11)
    Mux(row.orR, mixed, 0.U)
  }

  shadow.io.clock        := clock
  shadow.io.readEnable   := io.write.fire
  shadow.io.readAddress  := io.write.bits.addr
  shadow.io.writeEnable  := pending && !io.bind
  shadow.io.writeAddress := write.addr
  val readRow      = shadow.io.readData
  val readValidity = validityWords.read(io.write.bits.addr >> 5, io.write.fire)
  val oldValidity  = Mux(forwardGroup, previousValidity, Mux(groupWasValid, readValidity, 0.U))
  val rowBit       = UIntToOH(write.addr.pad(5)(4, 0), 32)
  val oldValid     = (oldValidity & rowBit).orR
  val nextValidity = oldValidity | rowBit
  when(pending && !io.bind) {
    validityWords.write(write.addr >> 5, nextValidity)
  }
  val oldRow       = Mux(oldValid, Mux(forward, previousRow, readRow), 0.U)
  val oldBytes     = oldRow.asTypeOf(Vec(16, UInt(8.W)))
  val writeBytes   = write.data.asTypeOf(Vec(16, UInt(8.W)))
  val newRow       = VecInit((0 until 16).map(i => Mux(write.mask(i), writeBytes(i), oldBytes(i)))).asUInt
  val nextHash     = statusHash - rowHash(write.addr, oldRow) + rowHash(write.addr, newRow)
  shadow.io.writeData := newRow

  io.write.ready := !io.bind
  // The pending write is already visible, matching the original write-edge contract.
  io.statusHash  := Mux(pending, nextHash, statusHash)

  when(io.bind) {
    statusHash  := 0.U
    validGroups := 0.U
    pending     := false.B
  }.otherwise {
    pending := io.write.fire
    when(io.write.fire) {
      val sameRow = pending && write.addr === io.write.bits.addr
      write         := io.write.bits
      groupWasValid := validGroups(io.write.bits.addr >> 5)
      forwardGroup  := pending && (write.addr >> 5) === (io.write.bits.addr >> 5)
      forward       := sameRow
    }
    when(pending) {
      validGroups      := validGroups | UIntToOH(write.addr >> 5, validityGroups)
      previousValidity := nextValidity
      statusHash       := nextHash
      previousRow      := newRow
    }
  }
}
