package framework.memdomain.midend

import chisel3._
import chisel3.util._
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
    val id     = UInt(log2Ceil(math.max(1, math.max(totalBallRead, totalBallWrite))).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(numChannels)(0.U.asTypeOf(new MappingTableEntry))))

  // -----------------------------------------------------------------------------
  // Core logic: allocate + same-cycle bypass routing
  // -----------------------------------------------------------------------------
  private val chSelW = math.max(1, log2Ceil(numChannels))

  // Free mask at cycle start
  val freeMask0 = VecInit(mappingTable.map(!_.valid)).asUInt

  // Detect existing mappings (logical port -> physical channel)
  val readMapped   = Wire(Vec(totalBallRead, Bool()))
  val readMappedCh = Wire(Vec(totalBallRead, UInt(chSelW.W)))
  for (rid <- 0 until totalBallRead) {
    val hits = VecInit((0 until numChannels).map { ch =>
      mappingTable(ch).valid && mappingTable(ch).isRead && (mappingTable(ch).id === rid.U)
    })
    readMapped(rid) := hits.asUInt.orR
    if (numChannels == 1) readMappedCh(rid) := 0.U else readMappedCh(rid) := OHToUInt(hits)
  }

  val writeMapped   = Wire(Vec(totalBallWrite, Bool()))
  val writeMappedCh = Wire(Vec(totalBallWrite, UInt(chSelW.W)))
  for (wid <- 0 until totalBallWrite) {
    val hits = VecInit((0 until numChannels).map { ch =>
      mappingTable(ch).valid && !mappingTable(ch).isRead && (mappingTable(ch).id === wid.U)
    })
    writeMapped(wid) := hits.asUInt.orR
    if (numChannels == 1) writeMappedCh(wid) := 0.U else writeMappedCh(wid) := OHToUInt(hits)
  }

  // Allocate unmapped requests onto free channels (combinational chaining, deterministic priority)
  val readAlloc   = Wire(Vec(totalBallRead, Bool()))
  val readAllocCh = Wire(Vec(totalBallRead, UInt(chSelW.W)))
  for (i <- 0 until totalBallRead) { readAlloc(i) := false.B; readAllocCh(i) := 0.U }

  val writeAlloc   = Wire(Vec(totalBallWrite, Bool()))
  val writeAllocCh = Wire(Vec(totalBallWrite, UInt(chSelW.W)))
  for (i <- 0 until totalBallWrite) { writeAlloc(i) := false.B; writeAllocCh(i) := 0.U }

  var freeMask = freeMask0

  for (rid <- 0 until totalBallRead) {
    val canAlloc = io.balldomain.bankRead(rid).io.req.valid && !readMapped(rid) && freeMask.orR
    val chosenCh = if (numChannels == 1) 0.U(chSelW.W) else PriorityEncoder(freeMask)
    readAlloc(rid)   := canAlloc
    readAllocCh(rid) := chosenCh
    freeMask = Mux(canAlloc, freeMask & ~(1.U(numChannels.W) << chosenCh), freeMask)
  }

  for (wid <- 0 until totalBallWrite) {
    val canAlloc = io.balldomain.bankWrite(wid).io.req.valid && !writeMapped(wid) && freeMask.orR
    val chosenCh = if (numChannels == 1) 0.U(chSelW.W) else PriorityEncoder(freeMask)
    writeAlloc(wid)   := canAlloc
    writeAllocCh(wid) := chosenCh
    freeMask = Mux(canAlloc, freeMask & ~(1.U(numChannels.W) << chosenCh), freeMask)
  }

  // -----------------------------------------------------------------------------
  // Backend Connection (static LHS, no dynamic LHS assigns)
  // -----------------------------------------------------------------------------
  for (ch <- 0 until numChannels) {
    io.mem_req(ch).read.req.valid   := false.B
    io.mem_req(ch).read.req.bits    := DontCare
    io.mem_req(ch).read.resp.ready  := true.B
    io.mem_req(ch).write.req.valid  := false.B
    io.mem_req(ch).write.req.bits   := DontCare
    io.mem_req(ch).write.resp.ready := true.B
    io.mem_req(ch).bank_id          := 0.U
  }

  // Default resp to ports (each port gets muxed resp from its selected channel)
  for (rid <- 0 until totalBallRead) {
    io.balldomain.bankRead(rid).io.resp.valid := false.B
    io.balldomain.bankRead(rid).io.resp.bits  := DontCare
  }
  for (wid <- 0 until totalBallWrite) {
    io.balldomain.bankWrite(wid).io.resp.valid := false.B
    io.balldomain.bankWrite(wid).io.resp.bits  := DontCare
  }

  // Per-port selection: mapped channel wins; otherwise same-cycle allocated channel
  val readActive = Wire(Vec(totalBallRead, Bool()))
  val readSelCh  = Wire(Vec(totalBallRead, UInt(chSelW.W)))
  for (rid <- 0 until totalBallRead) {
    readActive(rid) := readMapped(rid) || readAlloc(rid)
    readSelCh(rid)  := Mux(readMapped(rid), readMappedCh(rid), readAllocCh(rid))

    val mappedReady = io.mem_req(readMappedCh(rid)).read.req.ready
    val allocReady  = io.mem_req(readAllocCh(rid)).read.req.ready
    io.balldomain.bankRead(rid).io.req.ready := Mux(
      readMapped(rid),
      mappedReady,
      Mux(readAlloc(rid), allocReady, false.B)
    )

    io.balldomain.bankRead(rid).io.resp.valid := readActive(rid) && io.mem_req(readSelCh(rid)).read.resp.valid
    io.balldomain.bankRead(rid).io.resp.bits  := io.mem_req(readSelCh(rid)).read.resp.bits
  }

  val writeActive = Wire(Vec(totalBallWrite, Bool()))
  val writeSelCh  = Wire(Vec(totalBallWrite, UInt(chSelW.W)))
  for (wid <- 0 until totalBallWrite) {
    writeActive(wid) := writeMapped(wid) || writeAlloc(wid)
    writeSelCh(wid)  := Mux(writeMapped(wid), writeMappedCh(wid), writeAllocCh(wid))

    val mappedReady = io.mem_req(writeMappedCh(wid)).write.req.ready
    val allocReady  = io.mem_req(writeAllocCh(wid)).write.req.ready
    io.balldomain.bankWrite(wid).io.req.ready := Mux(
      writeMapped(wid),
      mappedReady,
      Mux(writeAlloc(wid), allocReady, false.B)
    )

    io.balldomain.bankWrite(wid).io.resp.valid := writeActive(wid) && io.mem_req(writeSelCh(wid)).write.resp.valid
    io.balldomain.bankWrite(wid).io.resp.bits  := io.mem_req(writeSelCh(wid)).write.resp.bits
  }

  // Per-channel serving: mapped first; if unmapped, allow same-cycle allocated request to bypass
  val chServeReadMapped  = Wire(Vec(numChannels, Bool()))
  val chServeWriteMapped = Wire(Vec(numChannels, Bool()))
  for (ch <- 0 until numChannels) {
    chServeReadMapped(ch)  := mappingTable(ch).valid && mappingTable(ch).isRead
    chServeWriteMapped(ch) := mappingTable(ch).valid && !mappingTable(ch).isRead
  }

  val chServeReadAlloc  = Wire(Vec(numChannels, Bool()))
  val chServeWriteAlloc = Wire(Vec(numChannels, Bool()))
  val chAllocReadId     = Wire(Vec(numChannels, UInt(log2Ceil(math.max(1, totalBallRead)).W)))
  val chAllocWriteId    = Wire(Vec(numChannels, UInt(log2Ceil(math.max(1, totalBallWrite)).W)))
  for (ch <- 0 until numChannels) {
    chServeReadAlloc(ch)  := false.B
    chServeWriteAlloc(ch) := false.B
    chAllocReadId(ch)     := 0.U
    chAllocWriteId(ch)    := 0.U
  }

  for (rid <- 0 until totalBallRead) {
    when(readAlloc(rid)) {
      chServeReadAlloc(readAllocCh(rid)) := true.B
      chAllocReadId(readAllocCh(rid))    := rid.U
    }
  }
  for (wid <- 0 until totalBallWrite) {
    when(writeAlloc(wid)) {
      chServeWriteAlloc(writeAllocCh(wid)) := true.B
      chAllocWriteId(writeAllocCh(wid))    := wid.U
    }
  }

  for (ch <- 0 until numChannels) {
    val serveMapped = mappingTable(ch).valid
    val serveRead   = chServeReadMapped(ch) || (!serveMapped && chServeReadAlloc(ch))
    val serveWrite  = chServeWriteMapped(ch) || (!serveMapped && chServeWriteAlloc(ch))

    when(serveRead) {
      val ridSel = Mux(chServeReadMapped(ch), mappingTable(ch).id, chAllocReadId(ch))

      io.mem_req(ch).read.req.valid  := io.balldomain.bankRead(ridSel).io.req.valid
      io.mem_req(ch).read.req.bits   := io.balldomain.bankRead(ridSel).io.req.bits
      io.mem_req(ch).read.resp.ready := io.balldomain.bankRead(ridSel).io.resp.ready
      io.mem_req(ch).bank_id         := io.balldomain.bankRead(ridSel).bank_id
    }.elsewhen(serveWrite) {
      val widSel = Mux(chServeWriteMapped(ch), mappingTable(ch).id, chAllocWriteId(ch))

      io.mem_req(ch).write.req.valid  := io.balldomain.bankWrite(widSel).io.req.valid
      io.mem_req(ch).write.req.bits   := io.balldomain.bankWrite(widSel).io.req.bits
      io.mem_req(ch).write.resp.ready := io.balldomain.bankWrite(widSel).io.resp.ready
      io.mem_req(ch).bank_id          := io.balldomain.bankWrite(widSel).bank_id
    }
  }

  // Commit mappingTable only when request actually fires into backend
  for (rid <- 0 until totalBallRead) {
    when(readAlloc(rid) && io.balldomain.bankRead(rid).io.req.fire) {
      val ch = readAllocCh(rid)
      mappingTable(ch).valid  := true.B
      mappingTable(ch).isRead := true.B
      mappingTable(ch).id     := rid.U
    }
  }
  for (wid <- 0 until totalBallWrite) {
    when(writeAlloc(wid) && io.balldomain.bankWrite(wid).io.req.fire) {
      val ch = writeAllocCh(wid)
      mappingTable(ch).valid  := true.B
      mappingTable(ch).isRead := false.B
      mappingTable(ch).id     := wid.U
    }
  }

  // Connect frontend to the last channel
  val frontendCh = b.top.memBallChannelNum
  io.mem_req(frontendCh).write <> io.frontend.bankWrite.io
  io.mem_req(frontendCh).read <> io.frontend.bankRead.io
  io.mem_req(frontendCh).bank_id := Mux(
    io.frontend.bankRead.io.req.valid,
    io.frontend.bankRead.bank_id,
    io.frontend.bankWrite.bank_id
  )

  // Mapping table release
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
    }.otherwise {
      releaseCounter := 0.U
    }
  }
}