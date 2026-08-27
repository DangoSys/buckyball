package framework.memdomain.midend

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.{MTraceIssueDPI, MemRequestIO}

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
    val bankRead          = Vec(totalRead, new BankReadWithShared(b))
    val bankWrite         = Vec(totalWrite, new BankWriteWithShared(b))
    val ballChannelActive = Input(Vec(b.ballDomain.ballNum, Bool()))
    val ballChannelReady  = Output(Vec(b.ballDomain.ballNum, Bool()))

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
    val isBall = Bool()
    val ballId = UInt(log2Up(b.ballDomain.ballNum).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(0.U.asTypeOf(new MappingTableEntry))))

  def isAllocated(isRead: Bool, id: UInt): Bool =
    mappingTable.map(entry => entry.valid && entry.isRead === isRead && entry.id === id).reduce(_ || _)

  val readPortBall = b.ballDomain.ballIdMappings.zipWithIndex.flatMap { case (mapping, ball) =>
    Seq.fill(mapping.inBW)(ball)
  }

  val writePortBall = b.ballDomain.ballIdMappings.zipWithIndex.flatMap { case (mapping, ball) =>
    Seq.fill(mapping.outBW)(ball)
  }

  for (ball <- 0 until b.ballDomain.ballNum) {
    val readReady  = readPortBall.zipWithIndex
      .filter(_._1 == ball)
      .map { case (_, port) => isAllocated(true.B, port.U) }
      .foldLeft(true.B)(_ && _)
    val writeReady = writePortBall.zipWithIndex
      .filter(_._1 == ball)
      .map { case (_, port) => isAllocated(false.B, port.U) }
      .foldLeft(true.B)(_ && _)
    io.ballChannelReady(ball) := io.ballChannelActive(ball) && readReady && writeReady
  }

  // Allocate exactly one channel per cycle. Ball channels are allocated before
  // their data traffic can start; frontend DMA channels are demand-allocated.
  for (i <- 0 until totalRead) {
    io.bankRead(i).bankRead.io.req.ready  := false.B
    io.bankRead(i).bankRead.io.resp.valid := false.B
    io.bankRead(i).bankRead.io.resp.bits  := DontCare

  }

  for (i <- 0 until totalWrite) {
    io.bankWrite(i).bankWrite.io.req.ready  := false.B
    io.bankWrite(i).bankWrite.io.resp.valid := false.B
    io.bankWrite(i).bankWrite.io.resp.bits  := DontCare
  }

  val pendingReads = VecInit((0 until totalRead).map { i =>
    val active =
      if (i < totalBallRead) {
        io.ballChannelActive(readPortBall(i))
      } else {
        io.bankRead(i).bankRead.io.req.valid
      }
    active && !isAllocated(true.B, i.U)
  })

  val pendingWrites = VecInit((0 until totalWrite).map { i =>
    val active =
      if (i < totalBallWrite) {
        io.ballChannelActive(writePortBall(i))
      } else {
        io.bankWrite(i).bankWrite.io.req.valid
      }
    active && !isAllocated(false.B, i.U)
  })

  val pending      = VecInit(pendingReads ++ pendingWrites)
  val freeChannels = mappingTable.map(entry => !entry.valid)

  when(pending.asUInt.orR && freeChannels.reduce(_ || _)) {
    val channel = PriorityEncoder(freeChannels)
    val port    = PriorityEncoder(pending)
    mappingTable(channel).valid  := true.B
    mappingTable(channel).isRead := port < totalRead.U
    mappingTable(channel).id     := Mux(port < totalRead.U, port, port - totalRead.U)
    mappingTable(channel).isBall := Mux(
      port < totalRead.U,
      MuxLookup(port, false.B)((0 until totalBallRead).map(i => i.U -> true.B).toSeq),
      MuxLookup(port - totalRead.U, false.B)((0 until totalBallWrite).map(i => i.U -> true.B).toSeq)
    )
    mappingTable(channel).ballId := Mux(
      port < totalRead.U,
      MuxLookup(port, 0.U)(readPortBall.zipWithIndex.map { case (ball, i) => i.U -> ball.U }.toSeq),
      MuxLookup(port - totalRead.U, 0.U)(writePortBall.zipWithIndex.map { case (ball, i) => i.U -> ball.U }.toSeq)
    )
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
    io.mem_req(i).rob_id           := 0.U

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
    val rrob_id   = io.bankRead(rid).bankRead.rob_id
    val wrob_id   = io.bankWrite(wid).bankWrite.rob_id

    when(mappingTable(i).valid) {
      when(isRead) {
        when(!mappingTable(i).isBall || io.ballChannelReady(mappingTable(i).ballId)) {
          io.mem_req(i).read <> ballRead
          io.mem_req(i).bank_id   := rbank_id
          io.mem_req(i).group_id  := rgroup_id
          io.mem_req(i).is_shared := r_shared
          io.mem_req(i).rob_id    := rrob_id
        }
      }.otherwise {
        when(!mappingTable(i).isBall || io.ballChannelReady(mappingTable(i).ballId)) {
          io.mem_req(i).write <> ballWrite
          io.mem_req(i).bank_id   := wbank_id
          io.mem_req(i).group_id  := wgroup_id
          io.mem_req(i).is_shared := w_shared
          io.mem_req(i).rob_id    := wrob_id
        }
      }
    }
  }

  // Count the actual write beats accepted by each backend channel. This is the
  // generation-side event at the same granularity as the target SRAM `fire`
  // arrival event, while remaining upstream of the backend/physical-bank path.
  // A DPI instance per channel preserves simultaneous accepted beats.
  val writeIssueTraces = Seq.fill(b.memDomain.bankChannel)(Module(new MTraceIssueDPI))
  for (i <- 0 until b.memDomain.bankChannel) {
    writeIssueTraces(i).io.clock     := clock
    writeIssueTraces(i).io.reset     := reset.asBool
    writeIssueTraces(i).io.hart_id   := io.mem_req(i).hart_id
    writeIssueTraces(i).io.is_shared := io.mem_req(i).is_shared.asUInt
    writeIssueTraces(i).io.rob_id    := io.mem_req(i).rob_id
    writeIssueTraces(i).io.vbank_id  := io.mem_req(i).bank_id
    writeIssueTraces(i).io.group_id  := io.mem_req(i).group_id
    writeIssueTraces(i).io.enable    := io.mem_req(i).write.req.fire
  }

  // Mapping table release
  for (i <- 0 until b.memDomain.bankChannel) {
    val releaseCounter = RegInit(0.U(5.W))

    when(mappingTable(i).valid && mappingTable(i).isBall &&
      !io.ballChannelActive(mappingTable(i).ballId)) {
      mappingTable(i).valid  := false.B
      mappingTable(i).isRead := false.B
      mappingTable(i).id     := 0.U
      mappingTable(i).isBall := false.B
      mappingTable(i).ballId := 0.U
      releaseCounter         := 0.U
    }.elsewhen(mappingTable(i).valid && !mappingTable(i).isBall && !(io.mem_req(i).read.resp.valid ||
      io.mem_req(i).write.resp.valid || io.mem_req(i).read.req.valid ||
      io.mem_req(i).write.req.valid)) {
      releaseCounter := releaseCounter + 1.U

      when(releaseCounter === 16.U) {
        releaseCounter         := 0.U
        mappingTable(i).valid  := false.B
        mappingTable(i).isRead := false.B
        mappingTable(i).id     := 0.U
        mappingTable(i).isBall := false.B
        mappingTable(i).ballId := 0.U
      }
    }.otherwise {
      releaseCounter := 0.U
    }
  }
}
