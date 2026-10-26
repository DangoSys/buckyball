package framework.device.peripherals.bootrom

import chisel3._
import framework.device.peripherals.bootrom.configs.BootRomParams
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.ip.bus.chi.{Params => ChiParams}
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

/**
 * The BootROM image (`bootrom/bootrom.rv64.img`, built from `bootrom.S`) with its two config dwords
 * filled: the hart count at offset 0x8 and the DRAM entry at offset 0x10.
 */
object BootRom {

  def apply(
    base:      BigInt,
    bytes:     BigInt,
    hartCount: Int,
    entry:     BigInt
  ): BootRomParams = {
    val stream = getClass.getResourceAsStream("/bootrom/bootrom.rv64.img")
    require(stream != null, "missing BootROM resource bootrom/bootrom.rv64.img")
    val image  =
      try stream.readAllBytes().toSeq
      finally stream.close()
    require(image.size >= 24 && image.slice(8, 24).forall(_ == 0), "BootROM image lacks its zeroed config dwords")
    def dword(value: BigInt): Seq[Byte] = (0 until 8).map(i => ((value >> (8 * i)) & 0xff).toByte)
    BootRomParams(base, bytes, image.take(8) ++ dword(hartCount) ++ dword(entry) ++ image.drop(24))
  }

}

/**
 * Serves read-only BootROM cache lines. Address routing belongs to the enclosing design. The
 * ROM is a cacheable, executable, read-only region, so fetches and loads reach it as line reads;
 * a write to a ROM line can only come from a faulty requester and answers with an error.
 */
@instantiable
class BootRomLines(p: BootRomParams, chi: ChiParams) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new LineRequest(chi)))
    val response = Decoupled(new LineResponse(chi))
  })

  val lines = p.image.padTo(((p.image.size + 63) / 64) * 64, 0.toByte).grouped(64).map { line =>
    line.zipWithIndex.map { case (byte, i) => BigInt(byte & 0xff) << (8 * i) }.sum
  }.toSeq

  val rom = VecInit(lines.map(_.U(512.W)))

  val offset  = io.request.bits.addr - p.base.U
  val inRom   = io.request.bits.addr >= p.base.U && io.request.bits.addr < (p.base + p.bytes).U
  val line    = offset >> 6
  val pending = RegInit(false.B)
  val id      = Reg(UInt(chi.txnIdBits.W))
  val data    = Reg(UInt(512.W))
  val error   = Reg(Bool())

  io.request.ready := !pending
  when(io.request.fire) {
    pending := true.B
    id      := io.request.bits.id
    error   := !inRom || io.request.bits.write || io.request.bits.addr(5, 0).orR
    data    := Mux(
      inRom && !io.request.bits.write && line < lines.size.U,
      rom(line(log2Ceil(lines.size.max(2)) - 1, 0)),
      0.U
    )
  }

  io.response.valid              := pending
  io.response.bits.id            := id
  io.response.bits.data          := Mux(error, 0.U, data)
  io.response.bits.error         := error
  when(io.response.fire)(pending := false.B)
}
