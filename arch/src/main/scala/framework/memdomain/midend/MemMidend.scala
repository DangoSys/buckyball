package framework.memdomain.midend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.MemRequestIO

@instantiable
class MemMidend(val b: GlobalConfig) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val numChannels    = b.memDomain.bankChannel - 1 // Reserve one for frontend

  @public
  val io = IO(new Bundle {

    val frontend = new Bundle {
      val bankRead  = new BankRead(b)
      val bankWrite = new BankWrite(b)
    }

    val balldomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))
    }

    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid  = Bool()
    val isRead = Bool()
    val id     = UInt(log2Ceil(math.max(totalBallRead, totalBallWrite)).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(numChannels)(0.U.asTypeOf(new MappingTableEntry))))

  def isAllocated(isRead: Bool, id: UInt): Bool =
    mappingTable.map(entry => entry.valid && entry.isRead === isRead && entry.id === id).reduce(_ || _)

  // -----------------------------------------------------------------------------
  // Core logic: multi-channel parallel allocation
  // -----------------------------------------------------------------------------

  // 1. Get which channels are free at cycle start (Bitmap)
  // use wire to propagate allocation within the cycle
  val initialFreeMask = VecInit(mappingTable.map(!_.valid)).asUInt

  var currentFreeMask = initialFreeMask

  // Define request object collection for unified iteration
  // Contains: (ReqValid, IsRead, ID, ReqReadyPtr)
  case class RequestCandidate(
    valid:  Bool,
    isRead: Bool,
    id:     Int,
    ready:  Bool)

  val candidates = scala.collection.mutable.ArrayBuffer[RequestCandidate]()

  // Collect read requests
  for (i <- 0 until totalBallRead) {
    // Default value
    io.balldomain.bankRead(i).io.req.ready  := !mappingTable.map(_.valid).reduce(_ || _)
    io.balldomain.bankRead(i).io.resp.valid := false.B
    io.balldomain.bankRead(i).io.resp.bits  := DontCare

    candidates += RequestCandidate(
      io.balldomain.bankRead(i).io.req.valid,
      true.B,
      i,
      io.balldomain.bankRead(i).io.req.ready
    )
  }

  // Collect write requests
  for (i <- 0 until totalBallWrite) {
    io.balldomain.bankWrite(i).io.req.ready  := !mappingTable.map(_.valid).reduce(_ || _)
    io.balldomain.bankWrite(i).io.resp.valid := false.B
    io.balldomain.bankWrite(i).io.resp.bits  := DontCare

    candidates += RequestCandidate(
      io.balldomain.bankWrite(i).io.req.valid,
      false.B,
      i,
      io.balldomain.bankWrite(i).io.req.ready
    )

  }

  for (cand <- candidates) {
    val needsAlloc = cand.valid && !isAllocated(cand.isRead, cand.id.U)
    val hasSpace   = currentFreeMask.orR // 检查当前掩码中是否还有1

    when(needsAlloc && hasSpace) {
      // Find lowest '1' in current mask (first free channel)
      val allocIdx = PriorityEncoder(currentFreeMask)

      mappingTable(allocIdx).valid  := true.B
      mappingTable(allocIdx).isRead := cand.isRead
      mappingTable(allocIdx).id     := cand.id.U

      cand.ready := true.B
    }

    // Key step: update mask, clear allocated bit
    // Logic generates hardware: next request sees current Mask minus allocIdx
    // Note: PriorityEncoder and shift operations are hardware logic
    val allocIdxWire = PriorityEncoder(currentFreeMask)
    // If needsAlloc is true, clear that bit; otherwise keep unchanged
    val nextMask     = Mux(needsAlloc && hasSpace, currentFreeMask & ~(1.U(numChannels.W) << allocIdxWire), currentFreeMask)

    currentFreeMask = nextMask
  }

  // -----------------------------------------------------------------------------
  // Backend Connection
  // -----------------------------------------------------------------------------
  for (i <- 0 until numChannels) {
    //Default values
    io.mem_req(i).read.req.valid   := false.B
    io.mem_req(i).read.req.bits    := DontCare
    io.mem_req(i).read.resp.ready  := true.B
    io.mem_req(i).write.req.valid  := false.B
    io.mem_req(i).write.req.bits   := DontCare
    io.mem_req(i).write.resp.ready := true.B
    io.mem_req(i).bank_id          := 0.U

    val entry = mappingTable(i)

    // Only connect when entry is valid
    when(entry.valid) {

      val ballRead  = io.balldomain.bankRead(entry.id).io
      val ballWrite = io.balldomain.bankWrite(entry.id).io
      val rbank_id  = io.balldomain.bankRead(entry.id).bank_id
      val wbank_id  = io.balldomain.bankWrite(entry.id).bank_id

      when(entry.isRead) {
        io.mem_req(i).read <> ballRead
        io.mem_req(i).bank_id := rbank_id
      }.otherwise {
        io.mem_req(i).write <> ballWrite
        io.mem_req(i).bank_id := wbank_id
      }
    }
  }

  // Connect frontend to the last channel
  val frontendCh = b.top.memBallChannelNum
  io.mem_req(frontendCh).write <> io.frontend.bankWrite.io
  io.mem_req(frontendCh).read <> io.frontend.bankRead.io
  io.mem_req(frontendCh).bank_id := io.frontend.bankWrite.bank_id

  //Mapping table release
  for (i <- 0 until numChannels) {
    val releaseCounter = RegInit(0.U(5.W))

    when(mappingTable(i).valid && !(io.mem_req(i).read.resp.valid ||
      io.mem_req(i).write.resp.valid || io.mem_req(i).read.req.valid ||
      io.mem_req(i).write.req.valid)) {
      releaseCounter := releaseCounter + 1.U

      when(releaseCounter === 16.U) {
        releaseCounter         := 0.U
        mappingTable(i).valid  := false.B
        mappingTable(i).isRead := false.B
        mappingTable(i).id     := 0.U
      }

    }
  }
}
