package examples.balls.vector

import framework.acc.balldomain.configs.BallRouteParam

import framework.lib.queue.Queue
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.ip.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.balls.vector.configs.VectorBallParam

class ctrl_st_req(frontendParam: FrontendParam, memParam: MemDomainParam, val ballParam: VectorBallParam)
    extends Bundle {
  val wr_bank      = UInt(frontendParam.bank_id_len.W)
  val wr_bank_addr = UInt(log2Up(memParam.bankEntries).W)
  val iter         = UInt(frontendParam.iter_len.W)
}

class ex_st_req(frontendParam: FrontendParam, val ballParam: VectorBallParam) extends Bundle {
  val config   = ballParam
  val InputNum = config.lane
  val accWidth = config.outputWidth
  val rst      = Vec(InputNum, UInt(accWidth.W))
  val iter     = UInt(frontendParam.iter_len.W)
}

class BankWriteEntry(memParam: MemDomainParam, val ballParam: VectorBallParam) extends Bundle {
  val addr = UInt(log2Ceil(memParam.bankEntries).W)
  val data = UInt(memParam.bankWidth.W)
  val mask = Vec(memParam.bankMaskLen, Bool())
}

@instantiable
class VecStoreUnit(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     VectorBallParam)
    extends Module {
  val config   = ballParam
  val InputNum = config.lane
  val accWidth = config.outputWidth

  // Use this instance's configured bandwidth.
  val ballMapping = instance

  val outBW = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new ctrl_st_req(frontendParam, memParam, ballParam)))
    val ex_st_i   = Flipped(Decoupled(new ex_st_req(frontendParam, ballParam)))
    val bankWrite = Vec(outBW, Flipped(new SramWriteIO(memParam)))
    val wr_bank_o = Output(UInt(frontendParam.bank_id_len.W))
    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  val wr_bank             = RegInit(0.U(frontendParam.bank_id_len.W))
  val wr_bank_addr        = RegInit(0.U(log2Up(memParam.bankEntries).W))
  val iter                = RegInit(0.U(frontendParam.iter_len.W))
  val iter_counter        = RegInit(0.U(frontendParam.iter_len.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  val writeQueues = VecInit(
    Seq.fill(outBW)(
      Instantiate(new Queue(new BankWriteEntry(memParam, ballParam), 16, false, false)).io
    )
  )

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := (io.ctrl_st_i.bits.iter + 15.U) & (~15.U(frontendParam.iter_len.W))
    iter_counter := 0.U
    state        := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit and push to write queues
// -----------------------------------------------------------------------------
  io.ex_st_i.ready := state === busy && writeQueues.forall(_.enq.ready)

  for (i <- 0 until outBW) {
    writeQueues(i).enq.valid := false.B
    writeQueues(i).enq.bits  := DontCare
  }

  when(io.ex_st_i.fire) {
    for (i <- 0 until outBW) {
      val elementsPerChannel = InputNum / outBW
      val startIdx           = i * elementsPerChannel
      val endIdx             = startIdx + elementsPerChannel - 1

      val entry = Wire(new BankWriteEntry(memParam, ballParam))
      entry.addr := wr_bank_addr + iter_counter(log2Ceil(InputNum) - 1, 0)
      entry.data := Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
      entry.mask := VecInit(Seq.fill(memParam.bankMaskLen)(true.B))

      writeQueues(i).enq.valid := true.B
      writeQueues(i).enq.bits  := entry
    }
    iter_counter := iter_counter + 1.U
  }

// -----------------------------------------------------------------------------
// Drain write queues to bankWrite interface
// -----------------------------------------------------------------------------
  io.bankWrite.foreach { acc =>
    acc.req.valid     := false.B
    acc.req.bits.addr := 0.U
    acc.req.bits.data := Cat(Seq.fill(accWidth / 8)(0.U(8.W)))
    acc.req.bits.mask := VecInit(Seq.fill(memParam.bankMaskLen)(false.B))
    acc.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    writeQueues(i).deq.ready := false.B

    when(writeQueues(i).deq.valid) {
      io.bankWrite(i).req.valid     := true.B
      io.bankWrite(i).req.bits.addr := writeQueues(i).deq.bits.addr
      io.bankWrite(i).req.bits.data := writeQueues(i).deq.bits.data
      io.bankWrite(i).req.bits.mask := writeQueues(i).deq.bits.mask
      writeQueues(i).deq.ready      := io.bankWrite(i).req.ready
    }
  }

  // Output wr_bank for bank_id setting
  io.wr_bank_o := wr_bank

// -----------------------------------------------------------------------------
// Reset iter counter, commit cmdResp, return to idle state
// -----------------------------------------------------------------------------
  val allQueuesEmpty  = writeQueues.forall(q => !q.deq.valid)
  val allDataEnqueued = state === busy && iter_counter >= iter

  when(allDataEnqueued && allQueuesEmpty) {
    state                    := idle
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.commit := false.B
  }

}
