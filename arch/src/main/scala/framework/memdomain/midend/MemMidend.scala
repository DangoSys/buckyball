package framework.memdomain.midend

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.MemRequestIO

// BankRead/BankWrite with is_shared flag, used for unified midend interface
class BankReadWithShared(val b: GlobalConfig) extends Bundle {
  val bankRead  = new BankRead(b)
  val is_shared = Input(Bool())
}

class BankWriteWithShared(val b: GlobalConfig) extends Bundle {
  val bankWrite = new BankWrite(b)
  val is_shared = Input(Bool())
}

/**
 * MemMidend: Midend module for memory scheduling
 * Connects MemFrontend to MemManager
 *
 * Unified interface: bankRead/bankWrite Vecs include both balldomain and frontend requests.
 * The last entry (index totalBallRead / totalBallWrite) is the frontend (DMA).
 * All requests go through the same mapping table and channel allocation logic.
 */
@instantiable
class MemMidend(val b: GlobalConfig) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  // Total slots: balldomain entries + 1 frontend entry
  val totalRead  = totalBallRead + 1
  val totalWrite = totalBallWrite + 1

  @public
  val io = IO(new Bundle {
    // Unified read/write interfaces: indices [0, totalBallRead) are balldomain,
    // index totalBallRead is frontend (DMA). Same for write.
    val bankRead  = Vec(totalRead, new BankReadWithShared(b))
    val bankWrite = Vec(totalWrite, new BankWriteWithShared(b))

    val hartid = Input(UInt(b.core.xLen.W))

    // Output to backend (MemManager)
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Mapping table for tracking all requests (balldomain + frontend)
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid  = Bool()
    val isRead = Bool()
    val id     = UInt(log2Ceil(math.max(totalRead, totalWrite)).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(0.U.asTypeOf(new MappingTableEntry))))

  def addEntry(idx: UInt, isRead: Bool, id: UInt): Unit = {
    mappingTable(idx).valid  := true.B
    mappingTable(idx).isRead := isRead
    mappingTable(idx).id     := id
  }

  def allocateChannel(): (Bool, UInt) = {
    val freeChannels = mappingTable.map(entry => !entry.valid)
    val hasFreeChan  = freeChannels.reduce(_ || _)
    val chanId       = PriorityEncoder(freeChannels)
    (hasFreeChan, chanId)
  }

  def isAllocated(isRead: Bool, id: UInt): Bool =
    mappingTable.map(entry => entry.valid && entry.isRead === isRead && entry.id === id).reduce(_ || _)

  // Allocate channels for reads (all entries including frontend)
  for (i <- 0 until totalRead) {
    io.bankRead(i).bankRead.io.req.ready  := false.B
    io.bankRead(i).bankRead.io.resp.valid := false.B
    io.bankRead(i).bankRead.io.resp.bits  := DontCare

    when(io.bankRead(i).bankRead.io.req.valid && !isAllocated(true.B, i.U)) {
      val (hasFree, chanId) = allocateChannel()
      when(hasFree) {
        addEntry(chanId, true.B, i.U)
      }
    }
  }

  // Allocate channels for writes: one per cycle to avoid conflicts
  val pendingWrites =
    VecInit((0 until totalWrite).map(i => io.bankWrite(i).bankWrite.io.req.valid && !isAllocated(false.B, i.U)))
  val hasPendingWrite     = pendingWrites.asUInt.orR
  val nextWriteToAllocate = PriorityEncoder(pendingWrites)

  for (i <- 0 until totalWrite) {
    io.bankWrite(i).bankWrite.io.req.ready  := false.B
    io.bankWrite(i).bankWrite.io.resp.valid := false.B
    io.bankWrite(i).bankWrite.io.resp.bits  := DontCare

    when(hasPendingWrite && nextWriteToAllocate === i.U) {
      val (hasFree, chanId) = allocateChannel()
      when(hasFree) {
        addEntry(chanId, false.B, i.U)
      }
    }
  }

  // Connect mapped entries to backend channels
  for (i <- 0 until b.memDomain.bankChannel) {
    io.mem_req(i).read.req.valid   := false.B
    io.mem_req(i).read.req.bits    := DontCare
    io.mem_req(i).read.resp.ready  := false.B
    io.mem_req(i).write.req.valid  := false.B
    io.mem_req(i).write.req.bits   := DontCare
    io.mem_req(i).write.resp.ready := false.B
    io.mem_req(i).bank_id          := 0.U
    io.mem_req(i).group_id         := 0.U
    io.mem_req(i).is_shared        := false.B
    io.mem_req(i).hart_id          := io.hartid

    val isRead    = mappingTable(i).isRead
    val rid       = mappingTable(i).id
    val wid       = mappingTable(i).id
    val ballRead  = io.bankRead(rid).bankRead.io
    val ballWrite = io.bankWrite(wid).bankWrite.io
    val rbank_id  = io.bankRead(rid).bankRead.bank_id
    val wbank_id  = io.bankWrite(wid).bankWrite.bank_id
    val rgroup_id = io.bankRead(rid).bankRead.group_id
    val wgroup_id = io.bankWrite(wid).bankWrite.group_id
    val r_shared  = io.bankRead(rid).is_shared
    val w_shared  = io.bankWrite(wid).is_shared

    when(mappingTable(i).valid) {
      when(isRead) {
        io.mem_req(i).read <> ballRead
        io.mem_req(i).bank_id  := rbank_id
        io.mem_req(i).group_id := rgroup_id
        io.mem_req(i).is_shared := r_shared
      }.otherwise {
        io.mem_req(i).write <> ballWrite
        io.mem_req(i).bank_id  := wbank_id
        io.mem_req(i).group_id := wgroup_id
        io.mem_req(i).is_shared := w_shared
      }
    }
  }

  // Mapping table release
  for (i <- 0 until b.memDomain.bankChannel) {
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
