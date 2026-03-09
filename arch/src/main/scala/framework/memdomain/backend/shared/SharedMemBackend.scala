package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.{MTraceDPI, MemRequestIO}
import framework.memdomain.backend.accpipe.AccPipe
import framework.memdomain.backend.banks.SramBank
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig

// class SharedArbReq(b: GlobalConfig) extends Bundle {
//   val src_channel = UInt(log2Up(b.memDomain.bankChannel).W)
//   val is_write    = Bool()
//   val hart_id     = UInt(b.core.xLen.W)
//   val pbank_id    = UInt(log2Ceil(SharedMemLayout.TotalBank).W)
//   val group_id    = UInt(3.W)
//   val addr        = UInt(log2Ceil(b.memDomain.bankEntries).W)
//   val mask        = Vec(b.memDomain.bankMaskLen, Bool())
//   val data        = UInt(b.memDomain.bankWidth.W)
//   val wmode       = Bool()
// }

@instantiable
class SharedMemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel*4, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(8.W))
    val query_group_count = Output(UInt(4.W))
  })

  // Shared backend is sized for 4 harts, but currently only hart-0 channels are connected.
  private val activeHartCount    = 1
  private val activeChannelCount = b.memDomain.bankChannel * activeHartCount

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum*4)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel*4)(Instantiate(new AccPipe(b)))

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = Seq.fill(b.memDomain.bankChannel*4)(Module(new MTraceDPI))
  for (mt <- mtraces) {
    mt.io.is_write := 0.U
    mt.io.channel  := 0.U
    mt.io.vbank_id := 0.U
    mt.io.group_id := 0.U
    mt.io.addr     := 0.U
    mt.io.data_lo  := 0.U
    mt.io.data_hi  := 0.U
    mt.io.enable   := false.B
  }

  // -----------------------------------------------------------------------------
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    
    val valid    = Bool()
    val hart_id  = UInt(b.core.xLen.W)
    val vbank_id = UInt(5.W)
    val is_multi = Bool()
    val group_id = UInt(3.W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankNum*4)(0.U.asTypeOf(new MappingTableEntry))))

  def isAcc(hart_id: UInt, vbank_id: UInt): Bool =
    mappingTable.map(entry => entry.valid && (entry.vbank_id === vbank_id) && (entry.hart_id === hart_id) && entry.is_multi).reduce(_ || _)

  def addEntry(
    hart_id: UInt,
    vbank_id: UInt,
    pbank_id: UInt,
    is_multi: Bool,
    group_id: UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid    := true.B
    entry.hart_id  := hart_id
    entry.vbank_id := vbank_id
    entry.is_multi := is_multi
    entry.group_id := group_id
  }

  def deleteEntry(hart_id: UInt, vbank_id: UInt): Unit = {
    for (i <- 0 until b.memDomain.bankNum*4) {
      when(mappingTable(i).valid && mappingTable(i).vbank_id === vbank_id && mappingTable(i).hart_id === hart_id) {
        mappingTable(i).valid    := false.B
        mappingTable(i).vbank_id := 0.U
        mappingTable(i).is_multi := false.B
        mappingTable(i).group_id := 0.U
      }
    }
  }

  def getFreePbankId(): UInt = {
    val freePbankId = mappingTable.indexWhere(_.valid === false.B)
    freePbankId
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until b.memDomain.bankChannel*4) {
    accPipes(i).io.mem_req.write <> io.mem_req(i).write
    accPipes(i).io.mem_req.read <> io.mem_req(i).read
    accPipes(i).io.mem_req.bank_id   := io.mem_req(i).bank_id
    accPipes(i).io.mem_req.group_id  := io.mem_req(i).group_id
    accPipes(i).io.mem_req.is_shared := io.mem_req(i).is_shared
    accPipes(i).io.mem_req.hart_id   := io.mem_req(i).hart_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    accPipes(i).io.is_multi := isAcc(io.mem_req(i).hart_id, io.mem_req(i).bank_id)
  }

  io.config.ready := true.B

  banks.zipWithIndex.foreach {
    case (bank, _) =>
      bank.io.sramRead.req.valid  := false.B
      bank.io.sramRead.req.bits   := DontCare
      bank.io.sramRead.resp.ready := true.B

      bank.io.sramWrite.req.valid  := false.B
      bank.io.sramWrite.req.bits   := DontCare
      bank.io.sramWrite.resp.ready := true.B
  }

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  when(io.config.valid) {
    when(io.config.bits.alloc) {
      val pbank_id = getFreePbankId()
      addEntry(io.config.bits.hart_id, io.config.bits.vbank_id, pbank_id, io.config.bits.is_multi, io.config.bits.group_id)
    }.otherwise {
      deleteEntry(io.config.bits.hart_id, io.config.bits.vbank_id)
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  val groupCounts = mappingTable.map { entry =>
    val matches = entry.valid && (entry.vbank_id === io.query_vbank_id)
    val count   = Mux(entry.is_multi, entry.group_id + 1.U, 1.U)
    Mux(matches, count, 0.U)
  }

  io.query_group_count := groupCounts.reduce((a, b) => Mux(a > b, a, b))

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  private def emitTrace(
    ch:      Int,
    isWrite: UInt,
    addr:    UInt,
    dataLo:  UInt,
    dataHi:  UInt,
    en:      Bool
  ): Unit = {
    mtraces(ch).io.is_write := isWrite
    mtraces(ch).io.channel  := ch.U
    mtraces(ch).io.vbank_id := io.mem_req(ch).bank_id
    mtraces(ch).io.group_id := io.mem_req(ch).group_id
    mtraces(ch).io.addr     := addr
    mtraces(ch).io.data_lo  := dataLo
    mtraces(ch).io.data_hi  := dataHi
    mtraces(ch).io.enable   := en
  }

  for (i <- 0 until activeChannelCount) {
    val req_valid = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid

    // Memory trace: read request
    when(io.mem_req(i).read.req.fire) {
      emitTrace(i, 0.U, io.mem_req(i).read.req.bits.addr, 0.U, 0.U, true.B)
    }

    // Memory trace: write request
    when(io.mem_req(i).write.req.fire) {
      emitTrace(
        i,
        1.U,
        io.mem_req(i).write.req.bits.addr,
        io.mem_req(i).write.req.bits.data(63, 0),
        io.mem_req(i).write.req.bits.data(127, 64),
        true.B
      )
    }

    for (j <- 0 until b.memDomain.bankNum*4) {
      val hit_bank = mappingTable(j).valid &&
        (mappingTable(j).hart_id === io.mem_req(i).hart_id) &&
        (mappingTable(j).vbank_id === io.mem_req(i).bank_id) &&
        (!mappingTable(j).is_multi ||
          (mappingTable(j).is_multi && (mappingTable(j).group_id === io.mem_req(i).group_id)))

      val hold_one = RegNext(hit_bank && req_valid, init = false.B)

      when((hit_bank && req_valid) || hold_one) {
        banks(j).io.sramRead <> accPipes(i).io.sramRead
        banks(j).io.sramWrite <> accPipes(i).io.sramWrite
      }
    }
  }
}
