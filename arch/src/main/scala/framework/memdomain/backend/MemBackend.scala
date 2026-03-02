package framework.memdomain.backend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

// DPI-C BlackBox for memory trace
class MTraceDPI extends BlackBox with HasBlackBoxInline {
  val io = IO(new Bundle {
    val is_write = Input(UInt(8.W))
    val channel  = Input(UInt(32.W))
    val vbank_id = Input(UInt(32.W))
    val group_id = Input(UInt(32.W))
    val addr     = Input(UInt(32.W))
    val data_lo  = Input(UInt(64.W))
    val data_hi  = Input(UInt(64.W))
    val enable   = Input(Bool())
  })

  setInline("MTraceDPI.v",
    """
    |import "DPI-C" function void dpi_mtrace(
    |  input byte unsigned is_write,
    |  input int unsigned channel,
    |  input int unsigned vbank_id,
    |  input int unsigned group_id,
    |  input int unsigned addr,
    |  input longint unsigned data_lo,
    |  input longint unsigned data_hi
    |);
    |
    |module MTraceDPI(
    |  input [7:0] is_write,
    |  input [31:0] channel,
    |  input [31:0] vbank_id,
    |  input [31:0] group_id,
    |  input [31:0] addr,
    |  input [63:0] data_lo,
    |  input [63:0] data_hi,
    |  input enable
    |);
    |  always @(*) begin
    |    if (enable) begin
    |      dpi_mtrace(is_write, channel, vbank_id, group_id, addr, data_lo, data_hi);
    |    end
    |  end
    |endmodule
    """.stripMargin)
}

class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write        = Flipped(new SramWriteIO(b)) // midend sends write req into backend
  val read         = Flipped(new SramReadIO(b))  // midend sends read req into backend
  val bank_id      = Output(UInt(log2Up(b.memDomain.bankNum).W))
  val group_id = Output(UInt(3.W))
}

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id = Input(UInt(8.W))
    val query_group_count = Output(UInt(4.W))
  })

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = Seq.fill(b.memDomain.bankChannel)(Module(new MTraceDPI))
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
    val valid        = Bool()
    val vbank_id     = UInt(5.W)
    val is_multi       = Bool()
    val group_id = UInt(3.W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U.asTypeOf(new MappingTableEntry))))

  def isAcc(vbank_id: UInt): Bool =
    mappingTable.map(entry => entry.valid && (entry.vbank_id === vbank_id) && entry.is_multi).reduce(_ || _)

  def addEntry(
    vbank_id:     UInt,
    pbank_id:     UInt,
    is_multi:       Bool,
    group_id: UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid        := true.B
    entry.vbank_id     := vbank_id
    entry.is_multi       := is_multi
    entry.group_id := group_id
  }

  def deleteEntry(vbank_id: UInt): Unit = {
    for (i <- 0 until b.memDomain.bankNum) {
      when(mappingTable(i).valid && mappingTable(i).vbank_id === vbank_id) {
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

  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.mem_req.write <> io.mem_req(i).write
    accPipes(i).io.mem_req.read <> io.mem_req(i).read
    accPipes(i).io.mem_req.bank_id      := io.mem_req(i).bank_id
    accPipes(i).io.mem_req.group_id := io.mem_req(i).group_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    accPipes(i).io.is_multi := isAcc(io.mem_req(i).bank_id)
  }

  io.config.ready := true.B

  banks.zipWithIndex.foreach {
    case (bank, i) =>
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
      addEntry(io.config.bits.vbank_id, pbank_id, io.config.bits.is_multi, io.config.bits.group_id)
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  val groupCounts = mappingTable.map { entry =>
    val matches = entry.valid && (entry.vbank_id === io.query_vbank_id)
    val count = Mux(entry.is_multi, entry.group_id + 1.U, 1.U)
    Mux(matches, count, 0.U)
  }
  io.query_group_count := groupCounts.reduce((a, b) => Mux(a > b, a, b))

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  for (i <- 0 until b.memDomain.bankChannel) {
    val hasReq = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid

    // Memory trace: read request
    when(io.mem_req(i).read.req.fire) {
      mtraces(i).io.is_write := 0.U
      mtraces(i).io.channel  := i.U
      mtraces(i).io.vbank_id := io.mem_req(i).bank_id
      mtraces(i).io.group_id := io.mem_req(i).group_id
      mtraces(i).io.addr     := io.mem_req(i).read.req.bits.addr
      mtraces(i).io.data_lo  := 0.U
      mtraces(i).io.data_hi  := 0.U
      mtraces(i).io.enable   := true.B
    }

    // Memory trace: write request
    when(io.mem_req(i).write.req.fire) {
      mtraces(i).io.is_write := 1.U
      mtraces(i).io.channel  := i.U
      mtraces(i).io.vbank_id := io.mem_req(i).bank_id
      mtraces(i).io.group_id := io.mem_req(i).group_id
      mtraces(i).io.addr     := io.mem_req(i).write.req.bits.addr
      mtraces(i).io.data_lo  := io.mem_req(i).write.req.bits.data(63, 0)
      mtraces(i).io.data_hi  := io.mem_req(i).write.req.bits.data(127, 64)
      mtraces(i).io.enable   := true.B
    }

    for (j <- 0 until b.memDomain.bankNum) {

      val req_valid = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid
      val hit_bank  = mappingTable(j).valid && (mappingTable(j).vbank_id === io.mem_req(i).bank_id) &&
        (!mappingTable(j).is_multi ||
          (mappingTable(j).is_multi && (mappingTable(j).group_id === io.mem_req(i).group_id)))

      val hold_one = RegNext(hit_bank && req_valid, init = false.B)

      // Debug: print when write request comes in
      when(io.mem_req(i).write.req.valid && i.U < 4.U) {
        printf("[Backend] ch=%d write req: vbank_id=%d group_id=%d\n", i.U, io.mem_req(i).bank_id, io.mem_req(i).group_id)
        printf("[Backend]   pbank[%d]: valid=%d vbank=%d is_multi=%d group=%d hit=%d\n",
          j.U, mappingTable(j).valid, mappingTable(j).vbank_id, mappingTable(j).is_multi, mappingTable(j).group_id, hit_bank)
      }

      when((hit_bank && req_valid) || hold_one) {
        banks(j).io.sramRead <> accPipes(i).io.sramRead
        banks(j).io.sramWrite <> accPipes(i).io.sramWrite
      }
    }
  }

}
