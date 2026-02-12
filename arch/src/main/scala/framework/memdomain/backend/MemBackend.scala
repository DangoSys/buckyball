package framework.memdomain.backend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write        = Flipped(new SramWriteIO(b)) // midend sends write req into backend
  val read         = Flipped(new SramReadIO(b))  // midend sends read req into backend
  val bank_id      = Output(UInt(log2Up(b.memDomain.bankNum).W))
  val acc_group_id = Output(UInt(3.W))
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
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid        = Bool()
    val vbank_id     = UInt(5.W)
    val is_acc       = Bool()
    val acc_group_id = UInt(3.W)
    val channel_id   = UInt(log2Ceil(b.memDomain.bankChannel).W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U.asTypeOf(new MappingTableEntry))))

  def isAcc(vbank_id: UInt): Bool =
    mappingTable.map(entry => entry.vbank_id === vbank_id && entry.is_acc).reduce(_ || _)

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
    entry.channel_id   := 0.U
  }

  def getFreePbankId(): UInt = {
    val freePbankId = mappingTable.indexWhere(_.valid === false.B)
    freePbankId
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.mem_req <> io.mem_req(i)

    accPipes(i).io.sramRead.req.ready  := true.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := true.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    accPipes(i).io.is_acc := isAcc(io.mem_req(i).bank_id)
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
      addEntry(io.config.bits.vbank_id, pbank_id, io.config.bits.is_acc, io.config.bits.acc_group_id)
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
    }
  }

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------

  for (i <- 0 until b.memDomain.bankChannel) {
    val hasReq = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid
    for (j <- 0 until b.memDomain.bankNum) {

      when((hasReq || RegNext(hasReq)) &&
        mappingTable(j).valid && (mappingTable(j).vbank_id === accPipes(i).io.bank_id) &&
        (!mappingTable(j).is_acc ||
          mappingTable(j).is_acc && (mappingTable(j).acc_group_id === accPipes(i).io.acc_group_id))) {

        mappingTable(j).channel_id := i.U
        banks(j).io.sramRead <> accPipes(i).io.sramRead
        banks(j).io.sramWrite <> accPipes(i).io.sramWrite
      }
    }
  }

}
