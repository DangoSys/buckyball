package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class BankPort(val b: GlobalConfig) extends Module {
  private val lineBytes  = b.memDomain.bankWidth / 8
  private val offsetBits = log2Ceil(lineBytes)
  require(b.memDomain.bankWidth >= 64 && isPow2(b.memDomain.bankWidth))
  require(lineBytes <= 65536)
  require(b.memDomain.bankMaskLen == lineBytes)

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new VectorMemoryRequest))
    val response = Decoupled(new VectorMemoryResponse)
    val read     = Flipped(new BankRead(b))
    val write    = Flipped(new BankWrite(b))
    val robId    = Input(UInt(log2Up(b.frontend.rob_entries).W))
    val ballId   = Input(UInt(log2Up(b.ballDomain.ballNum).W))
  })

  private val idle :: readRequest :: readResponse :: writeRequest :: writeResponse :: complete :: Nil = Enum(6)
  private val state                                                                                   = RegInit(idle)
  private val request                                                                                 = Reg(new VectorMemoryRequest)
  private val robId                                                                                   = Reg(UInt(log2Up(b.frontend.rob_entries).W))
  private val ballId                                                                                  = Reg(UInt(log2Up(b.ballDomain.ballNum).W))
  private val consumed                                                                                = RegInit(0.U(4.W))
  private val result                                                                                  = RegInit(0.U(64.W))
  private val error                                                                                   = RegInit(false.B)

  private val byteCount     = (1.U(4.W) << request.size)(3, 0)
  private val byteAddress   = request.address(15, 0) +& consumed
  private val lineOffset    = byteAddress(offsetBits - 1, 0)
  private val remaining     = byteCount - consumed
  private val lineRemaining = lineBytes.U - lineOffset
  private val chunk         = Mux(remaining <= lineRemaining, remaining, lineRemaining)(3, 0)
  private val nextConsumed  = consumed + chunk
  private val lineAddress   = byteAddress >> offsetBits
  private val readShifted   = io.read.io.resp.bits.data >> (lineOffset << 3)
  private val chunkMask     = ((1.U(65.W) << (chunk << 3)) - 1.U)(63, 0)
  private val readChunk     = readShifted.pad(64)(63, 0) & chunkMask

  io.request.ready          := state === idle
  io.response.valid         := state === complete
  io.response.bits.data     := result
  io.response.bits.error    := error
  io.read.bank_id           := request.address(31, 16)
  io.read.rob_id            := robId
  io.read.ball_id           := ballId
  io.read.group_id          := 0.U
  io.read.io.req.valid      := state === readRequest
  io.read.io.req.bits.addr  := lineAddress
  io.read.io.resp.ready     := state === readResponse
  io.write.bank_id          := request.address(31, 16)
  io.write.rob_id           := robId
  io.write.ball_id          := ballId
  io.write.group_id         := 0.U
  io.write.io.req.valid     := state === writeRequest
  io.write.io.req.bits.addr := lineAddress
  io.write.io.req.bits.data := (request.data >> (consumed << 3)).pad(b.memDomain.bankWidth) << (lineOffset << 3)
  io.write.io.resp.ready    := state === writeResponse
  for (i <- 0 until lineBytes) {
    val sourceByte = consumed + i.U - lineOffset
    io.write.io.req.bits.mask(i) := i.U >= lineOffset && i.U < lineOffset +& chunk && request.mask(sourceByte(2, 0))
  }

  when(io.request.fire) {
    request  := io.request.bits
    robId    := io.robId
    ballId   := io.ballId
    consumed := 0.U
    result   := 0.U
    val end     = io.request.bits.address(15, 0) +& (1.U(17.W) << io.request.bits.size)
    val invalid = io.request.bits.address(31, 16) > b.frontend.vbank_id_upper_bound.U ||
      end > (BigInt(b.memDomain.bankEntries) * lineBytes).U || end > 65536.U
    error := invalid
    state := Mux(invalid, complete, Mux(io.request.bits.write, writeRequest, readRequest))
  }
  when(io.read.io.req.fire) {
    state := readResponse
  }
  when(io.read.io.resp.fire) {
    result   := result | (readChunk << (consumed << 3))
    consumed := nextConsumed
    state    := Mux(nextConsumed === byteCount, complete, readRequest)
  }
  when(io.write.io.req.fire) {
    state := writeResponse
  }
  when(io.write.io.resp.fire) {
    consumed := nextConsumed
    error    := !io.write.io.resp.bits.ok
    state    := Mux(!io.write.io.resp.bits.ok || nextConsumed === byteCount, complete, writeRequest)
  }
  when(io.response.fire) {
    state := idle
  }
}
