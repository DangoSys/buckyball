package framework.memdomain.backend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write   = Flipped(new SramWriteIO(b)) // midend sends write req into backend
  val read    = Flipped(new SramReadIO(b))  // midend sends read req into backend
  val bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W))
}

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))
  })

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))

  // -----------------------------------------------------------------------------
  // Mapping table/ Free List
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid        = Bool()
    val vbank_id     = UInt(5.W)
    val is_acc       = Bool()
    val acc_group_id = UInt(3.W)
    val channel_id   = UInt(log2Ceil(b.memDomain.bankChannel).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U.asTypeOf(new MappingTableEntry))))

  def isAllocated(vbank_id: UInt): Bool =
    mappingTable.map(_.vbank_id === vbank_id).reduce(_ || _)

  def addEntry(
    vbank_id:     UInt,
    pbank_id:     UInt,
    is_acc:       Bool,
    acc_group_id: UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid        := true.B
    entry.vbank_id     := vbank_id
    entry.is_acc       := is_acc
    entry.acc_group_id := acc_group_id
  }

  def deleteEntry(vbank_id: UInt): Unit = {
    val entry = mappingTable(vbank_id)
    entry.valid        := false.B
    entry.vbank_id     := 0.U
    entry.is_acc       := false.B
    entry.acc_group_id := 0.U
  }

  def getFreePbankId(): UInt = {
    val freePbankId = mappingTable.indexWhere(_.valid === false.B)
    freePbankId
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.write <> io.mem_req(i).write
    accPipes(i).io.read <> io.mem_req(i).read
    accPipes(i).io.bank_id := io.mem_req(i).bank_id

    accPipes(i).io.sramRead.req.ready  := true.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := true.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    accPipes(i).io.is_acc := false.B
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

  when(io.config.valid && !io.config.bits.is_acc) {
    when(io.config.bits.alloc) {
      val pbank_id = getFreePbankId()
      addEntry(io.config.bits.vbank_id, pbank_id, io.config.bits.is_acc, 0.U)
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
    }
  }

  when(io.config.valid && io.config.bits.is_acc) {}

  for (i <- 0 until b.memDomain.bankChannel) {
    for (j <- 0 until b.memDomain.bankNum) {
      when((io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid) &&
        mappingTable(j).valid && (mappingTable(j).vbank_id === io.mem_req(i).bank_id)) {
        mappingTable(j).channel_id := i.U
      }
    }
  }

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  banks.zipWithIndex.foreach {
    case (bank, pBankId) =>
      accPipes.zipWithIndex.foreach {
        case (accPipe, accPipeId) =>
          when(mappingTable(pBankId).valid && (mappingTable(pBankId).channel_id === accPipeId.U)) {
            bank.io.sramRead <> accPipe.io.sramRead
            bank.io.sramRead.req.valid := RegNext(accPipe.io.sramRead.req.valid)
            bank.io.sramRead.req.bits  := RegNext(accPipe.io.sramRead.req.bits)

            bank.io.sramWrite <> accPipe.io.sramWrite
            bank.io.sramWrite.req.valid := RegNext(accPipe.io.sramWrite.req.valid)
            bank.io.sramWrite.req.bits  := RegNext(accPipe.io.sramWrite.req.bits)

          }
      }
  }
}
