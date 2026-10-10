package framework.system.device

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import memcore.bus.chi.{Params => ChiParams}
import memcore.bus.chi.snf.{LineRequest, LineResponse}

/**
 * The BootROM image (`bootrom/bootrom.rv64.img`, built from `bootrom.S`) with its two config dwords
 * filled: the hart count at offset 0x8 and the DRAM entry at offset 0x10.
 */
case class BootRomParams(base: BigInt, bytes: BigInt, image: Seq[Byte]) {
  require(base % 64 == 0 && bytes % 64 == 0 && image.size <= bytes, "BootROM must be whole cache lines")
}

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
 * Serves BootROM lines on a tile's L2 backing path and forwards every other line to memory. The
 * ROM is a cacheable, executable, read-only region, so fetches and loads reach it as line reads;
 * a write to a ROM line can only come from a faulty requester and answers with an error.
 */
@instantiable
class BootRomLines(p: BootRomParams, chi: ChiParams) extends Module {

  @public val io = IO(new Bundle {
    val request        = Flipped(Decoupled(new LineRequest(chi)))
    val response       = Decoupled(new LineResponse(chi))
    val memoryRequest  = Decoupled(new LineRequest(chi))
    val memoryResponse = Flipped(Decoupled(new LineResponse(chi)))
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

  io.memoryRequest.valid := io.request.valid && !inRom
  io.memoryRequest.bits  := io.request.bits
  io.request.ready       := Mux(inRom, !pending, io.memoryRequest.ready)
  when(io.request.fire && inRom) {
    pending := true.B
    id      := io.request.bits.id
    data    := Mux(line < lines.size.U, rom(line(log2Ceil(lines.size.max(2)) - 1, 0)), 0.U)
    error   := io.request.bits.write
  }

  // Memory responses keep priority; the ROM answer waits at most until memory is idle.
  io.response.valid                                          := io.memoryResponse.valid || pending
  io.response.bits                                           := Mux(io.memoryResponse.valid, io.memoryResponse.bits, 0.U.asTypeOf(io.response.bits))
  when(!io.memoryResponse.valid) {
    io.response.bits.id    := id
    io.response.bits.data  := data
    io.response.bits.error := error
  }
  io.memoryResponse.ready                                    := io.response.ready
  when(io.response.fire && !io.memoryResponse.valid)(pending := false.B)
}
