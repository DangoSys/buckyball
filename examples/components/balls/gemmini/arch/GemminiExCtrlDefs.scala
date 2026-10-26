package examples.balls.gemmini

import framework.lib.queue.Queue
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.ip.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import framework.acc.balldomain.ball.blink.BallStatus
import framework.ip.memory.bank.{SramReadReq, SramReadResp, SramWriteIO}
import examples.balls.gemmini.configs.GemminiBallParam
import gemmini._
import gemmini.Util._

/** Minimal tag for MeshWithDelays that satisfies TagQueueTag */
class SimpleTag extends Bundle with gemmini.TagQueueTag {
  val rob = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit =
    rob := 0xff.U
}

trait GemminiExCtrlDefs { this: GemminiExCtrl =>
  val config = ballParam
  val DIM    = config.blockSize

  val ballMapping = instance

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  val inputType      = SInt(config.inputWidth.W)
  val accType        = SInt(config.accWidth.W)
  val meshOutputType = SInt(config.spatialOutputWidth.W)

  val ctrlIo = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp      = Decoupled(new BallResponse(frontendParam))
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(memParam)))
    val bankReadResp =
      Vec(inBW, Flipped(Decoupled(new SramReadResp(memParam))))
    val bankWrite    = Vec(outBW, Flipped(new SramWriteIO(memParam)))
    val op1_bank_o   = Output(UInt(frontendParam.bank_id_len.W))
    val op1_group_o  = Output(UInt(memParam.groupIdWidth.W))
    val op2_bank_o   = Output(UInt(frontendParam.bank_id_len.W))
    val op2_group_o  = Output(UInt(memParam.groupIdWidth.W))
    val wr_bank_o    = Output(UInt(frontendParam.bank_id_len.W))
    val wr_group_o   = Output(UInt(memParam.groupIdWidth.W))
    val status       = new BallStatus
  })

  val io = ctrlIo

  val mesh = Module(new MeshWithDelays(
    inputType,
    inputType,
    meshOutputType,
    accType,
    new SimpleTag,
    Dataflow.BOTH,
    false,
    config.tileLatency,
    config.outputDelay,
    config.tileRows,
    config.tileColumns,
    config.meshRows,
    config.meshColumns,
    1,
    1
  ))

  protected def widenMeshToAcc(src: Vec[Vec[SInt]]): Vec[Vec[SInt]] =
    VecInit(src.map(col => VecInit(col.map(_.asTypeOf(accType)))))

  val cfg_dataflow       = RegInit(0.U(1.W))
  val cfg_in_shift       = RegInit(0.U(log2Up(config.accWidth).W))
  val cfg_a_transpose    = RegInit(false.B)
  val cfg_bd_transpose   = RegInit(false.B)
  val zero_op2           = RegInit(false.B)
  val zero_op1_tail      = RegInit(false.B)
  val accumulated_cmd    = RegInit(false.B)
  val reset_acc_mesh     = RegInit(false.B)
  val acc_read_group     = RegInit(0.U(log2Up(outBW).W))
  val acc_read_pending   = RegInit(false.B)
  val execution_dataflow = Mux(accumulated_cmd, Dataflow.OS.id.U, cfg_dataflow)
  val execution_shift    = Mux(accumulated_cmd, 0.U, cfg_in_shift)

  val sIdle :: sPreloadRead :: sPreloadFeed :: sComputeRead :: sComputeFeed :: sComputeFlush :: sFlush :: sDrain :: sReadAccum :: sStore :: sCommit :: Nil =
    Enum(11)
  val state                                                                                                                                                = RegInit(sIdle)

  val rob_id_reg = RegInit(0.U(log2Up(frontendParam.rob_entries).W))

  /** Match SimpleTag.rob (8b); must match mesh req tag bits */
  protected def robIdAsTag8(x: UInt): UInt = {
    val w = x.getWidth
    if (w >= 8) x(7, 0) else Cat(0.U((8 - w).W), x)
  }

  val op1_bank     = RegInit(0.U(frontendParam.bank_id_len.W))
  val op2_bank     = RegInit(0.U(frontendParam.bank_id_len.W))
  val wr_bank      = RegInit(0.U(frontendParam.bank_id_len.W))
  val wr_read_bank = RegInit(0.U(frontendParam.bank_id_len.W))
  val op1_group    = RegInit(0.U(memParam.groupIdWidth.W))
  val op2_group    = RegInit(0.U(memParam.groupIdWidth.W))
  val wr_group     = RegInit(0.U(memParam.groupIdWidth.W))
  val op1_base     = RegInit(0.U(10.W))
  val op2_base     = RegInit(0.U(10.W))
  val wr_base      = RegInit(0.U(10.W))
  io.op1_bank_o  := Mux(state === sReadAccum, wr_read_bank, op1_bank)
  io.op1_group_o := Mux(state === sReadAccum, wr_group + acc_read_group, op1_group)
  io.op2_bank_o  := op2_bank
  io.op2_group_o := op2_group
  io.wr_bank_o   := wr_bank
  io.wr_group_o  := wr_group

  val read_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val feed_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val store_row_cnt = RegInit(0.U(log2Up(DIM + 1).W))
  val total_rows    = RegInit(DIM.U(log2Up(DIM + 1).W))
  val req_sent      = RegInit(false.B)
  val read_done     = RegInit(VecInit(Seq.fill(inBW)(false.B)))
  val xpose_ready   = RegInit(false.B)
  val xpose_row_cnt = RegInit(0.U(log2Up(DIM + 1).W))

  val sub_cmd = io.cmdReq.bits.rs2(3, 0)

  val rdQueue0 = Instantiate(new Queue(new SramReadResp(memParam), DIM))
  val rdQueue1 = Instantiate(new Queue(new SramReadResp(memParam), DIM))
  rdQueue0.io.enq <> io.bankReadResp(0)
  rdQueue1.io.enq <> io.bankReadResp(1)

  val outBuf = Reg(
    Vec(DIM, Vec(config.meshColumns, Vec(config.tileColumns, accType)))
  )

  val outBufRows      = RegInit(0.U(log2Up(DIM + 1).W))
  val outBufCollected = RegInit(0.U(log2Up(DIM + 1).W))
  val op1Buf          = Reg(Vec(DIM, Vec(DIM, inputType)))
  val op2Buf          = Reg(Vec(DIM, Vec(DIM, inputType)))

  val port_written = RegInit(VecInit(Seq.fill(outBW)(false.B)))
}
