package framework.memdomain.midend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.MemRequestIO

/**
 * MemMidend: Midend module for memory scheduling
 * Connects MemFrontend to MemManager
 *
 * Basic direct connection: routes requests from frontend to backend channels
 */
@instantiable
class MemMidend(val b: GlobalConfig) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {

    // Input from frontend (Ball Domain read/write requests) - receiver perspective
    val frontend = new Bundle {
      val bankRead  = new BankRead(b)
      val bankWrite = new BankWrite(b)
    }

    val balldomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))
    }

    // Output to backend (MemManager) - MemManager expects Flipped, so we don't flip here
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Mapping table for tracking balldomain requests
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid  = Bool()
    val isRead = Bool()
    val id     = UInt(log2Ceil(math.max(totalBallRead, totalBallWrite)).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel - 1)(0.U.asTypeOf(new MappingTableEntry))))

  def addEntry(idx: UInt, isRead: Bool, id: UInt): Unit = {
    mappingTable(idx).valid  := true.B
    mappingTable(idx).isRead := isRead
    mappingTable(idx).id     := id
  }

  def allocateChannel(): UInt = {
    val freeChannels = mappingTable.map(entry => !entry.valid)
    PriorityEncoder(freeChannels)
  }

  def isAllocated(isRead: Bool, id: UInt): Bool =
    mappingTable.map(entry => entry.valid && entry.isRead === isRead && entry.id === id).reduce(_ || _)

  for (i <- 0 until totalBallRead) {
    //Default values
    io.balldomain.bankRead(i).io.req.ready  := false.B
    io.balldomain.bankRead(i).io.resp.valid := false.B;
    io.balldomain.bankRead(i).io.resp.bits  := DontCare

    when(io.balldomain.bankRead(i).io.req.valid && !isAllocated(true.B, i.U)) {
      addEntry(allocateChannel(), true.B, i.U)
    }
  }

  for (i <- 0 until totalBallWrite) {
    //Default values
    io.balldomain.bankWrite(i).io.req.ready  := false.B
    io.balldomain.bankWrite(i).io.resp.valid := false.B;
    io.balldomain.bankWrite(i).io.resp.bits  := DontCare
    //disable for now

    when(io.balldomain.bankWrite(i).io.req.valid && !isAllocated(false.B, i.U)) {
      addEntry(allocateChannel(), false.B, i.U)
      }
  }

  //Connect balldomain to backend
  for (i <- 0 until b.top.memBallChannelNum) {
    //Default values
    io.mem_req(i).read.req.valid   := false.B
    io.mem_req(i).read.req.bits    := DontCare
    io.mem_req(i).read.resp.ready  := false.B
    io.mem_req(i).write.req.valid  := false.B
    io.mem_req(i).write.req.bits   := DontCare
    io.mem_req(i).write.resp.ready := false.B
    io.mem_req(i).bank_id          := 0.U
    io.mem_req(i).group_id     := 0.U

    val isRead    = mappingTable(i).isRead
    val ballRead  = io.balldomain.bankRead(mappingTable(i).id).io
    val ballWrite = io.balldomain.bankWrite(mappingTable(i).id).io
    val rbank_id  = io.balldomain.bankRead(mappingTable(i).id).bank_id
    val wbank_id  = io.balldomain.bankWrite(mappingTable(i).id).bank_id
    val group_id    = io.balldomain.bankWrite(mappingTable(i).id).group_id

    when(mappingTable(i).valid) {
      when(isRead) {
        io.mem_req(i).read <> ballRead
        io.mem_req(i).bank_id := rbank_id
        io.mem_req(i).group_id := group_id
      }.otherwise {
        io.mem_req(i).write <> ballWrite
        io.mem_req(i).bank_id := wbank_id
        io.mem_req(i).group_id := group_id
      }
    }
  }

  //Connect frontend to backend
  io.mem_req(b.top.memBallChannelNum).write <> io.frontend.bankWrite.io
  io.mem_req(b.top.memBallChannelNum).read <> io.frontend.bankRead.io
    io.mem_req(b.top.memBallChannelNum).bank_id      := Mux(
    io.frontend.bankRead.io.req.valid,
    io.frontend.bankRead.bank_id,
    io.frontend.bankWrite.bank_id
  )
  io.mem_req(b.top.memBallChannelNum).group_id := Mux(
    io.frontend.bankRead.io.req.valid,
    io.frontend.bankRead.group_id,
    io.frontend.bankWrite.group_id
  )

  // Mapping table release
  for (i <- 0 until b.top.memBallChannelNum - 1) {
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
