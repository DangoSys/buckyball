package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.public
import framework.balldomain.blink.BallStatus
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp, SramWriteIO}
import framework.top.GlobalConfig
import framework.balldomain.prototype.gemmini.configs.GemminiBallParam
import gemmini._
import gemmini.Util._

trait GemminiExCtrlDefs { this: GemminiExCtrl =>
  val config = GemminiBallParam()
  val DIM    = config.blockSize

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "GemminiBall")
    .getOrElse(throw new IllegalArgumentException("GemminiBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  val inputType      = SInt(config.inputWidth.W)
  val accType        = SInt(config.accWidth.W)
  val meshOutputType = SInt(config.spatialOutputWidth.W)

  val ctrlIo = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))
    val bankWrite    = Vec(outBW, Flipped(new SramWriteIO(b)))
    val op1_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val op2_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wr_bank_o    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val status       = new BallStatus
  })

  val io = ctrlIo

  val mesh = Module(new MeshWithDelays(
    inputType = inputType,
    outputType = meshOutputType,
    accType = accType,
    tagType = new SimpleTag,
    df = Dataflow.BOTH,
    tree_reduction = false,
    tile_latency = config.tileLatency,
    output_delay = config.outputDelay,
    tileRows = config.tileRows,
    tileColumns = config.tileColumns,
    meshRows = config.meshRows,
    meshColumns = config.meshColumns,
    leftBanks = 1,
    upBanks = 1
  ))

  protected def widenMeshToAcc(src: Vec[Vec[SInt]]): Vec[Vec[SInt]] =
    VecInit(src.map(col => VecInit(col.map(_.asTypeOf(accType)))))

  val cfg_dataflow     = RegInit(0.U(1.W))
  val cfg_in_shift     = RegInit(0.U(log2Up(config.accWidth).W))
  val cfg_a_transpose  = RegInit(false.B)
  val cfg_bd_transpose = RegInit(false.B)

  val sIdle :: sPreloadRead :: sPreloadFeed :: sComputeRead :: sComputeFeed :: sComputeFlush :: sFlush :: sDrain :: sStore :: sCommit :: Nil =
    Enum(10)
  val state                                                                                                                                  = RegInit(sIdle)

  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  /** Match SimpleTag.rob (8b); must match mesh req tag bits */
  protected def robIdAsTag8(x: UInt): UInt = {
    val w = x.getWidth
    if (w >= 8) x(7, 0) else Cat(0.U((8 - w).W), x)
  }

  val op1_bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2_bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  io.op1_bank_o := op1_bank
  io.op2_bank_o := op2_bank
  io.wr_bank_o  := wr_bank

  val read_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val feed_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val store_row_cnt = RegInit(0.U(log2Up(DIM + 1).W))
  val total_rows    = RegInit(DIM.U(log2Up(DIM + 1).W))
  val req_sent      = RegInit(false.B)

  val sub_cmd = io.cmdReq.bits.cmd.special(3, 0)

  val rdQueue0 = Module(new Queue(new SramReadResp(b), entries = DIM))
  val rdQueue1 = Module(new Queue(new SramReadResp(b), entries = DIM))
  rdQueue0.io.enq <> io.bankReadResp(0)
  rdQueue1.io.enq <> io.bankReadResp(1)

  val outBuf          = Reg(Vec(DIM, Vec(config.meshColumns, Vec(config.tileColumns, accType))))
  val outBufRows      = RegInit(0.U(log2Up(DIM + 1).W))
  val outBufCollected = RegInit(0.U(log2Up(DIM + 1).W))

  val port_written = RegInit(VecInit(Seq.fill(outBW)(false.B)))
}
