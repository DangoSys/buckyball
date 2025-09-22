package examples.toy.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy.{LazyModule, LazyModuleImp}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.PostGDCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.rocket.RoCCResponseBB
import examples.toy.balldomain.rs.BallReservationStation
import examples.toy.balldomain.bbus.BBusModule

// Ball Domain的输入输出接口
class BallDomainIO(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // 来自GlobalDecoder的命令分发接口
  val gDecoderIn = Flipped(Decoupled(new PostGDCmd))

  // 连接到Scratchpad的执行接口
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

  // RoCC响应接口
  val roccResp = Decoupled(new RoCCResponseBB()(p))
  val busy = Output(Bool())

  // fence信号
  val fence_o = Output(Bool())
}

// Ball Domain 顶层 - 使用新的简化BBus架构
class BallDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {

  val io = IO(new BallDomainIO)

  // 创建新的BBus模块
  val bbus = Module(new BBusModule)

//---------------------------------------------------------------------------
// Decoder -> BallReservationStation
//---------------------------------------------------------------------------
  val ballDecoder = Module(new BallDomainDecoder)
  ballDecoder.io.raw_cmd_i <> io.gDecoderIn

  // fence信号连接
  io.fence_o := ballDecoder.io.fence_o

//---------------------------------------------------------------------------
// Decoder -> BallReservationStation
//---------------------------------------------------------------------------
  val ballRs = Module(new BallReservationStation)
  ballRs.io.ball_decode_cmd_i <> ballDecoder.io.ball_decode_cmd_o

//---------------------------------------------------------------------------
// BallReservationStation -> BBus (使用新的简化BBus)
//---------------------------------------------------------------------------
  // 创建命令请求仲裁器，将4个Ball的请求合并
  val cmdReqArbiter = Module(new RRArbiter(new examples.toy.balldomain.rs.BallRsIssue, 4))
  cmdReqArbiter.io.in(0) <> ballRs.io.issue_o.ball1
  cmdReqArbiter.io.in(1) <> ballRs.io.issue_o.ball2
  cmdReqArbiter.io.in(2) <> ballRs.io.issue_o.ball3
  cmdReqArbiter.io.in(3) <> ballRs.io.issue_o.ball4
  bbus.io.cmdReq <> cmdReqArbiter.io.out

  // 创建响应分发器，将BBus的响应广播给4个Ball
  ballRs.io.commit_i.ball1.valid := bbus.io.cmdResp.valid
  ballRs.io.commit_i.ball1.bits := bbus.io.cmdResp.bits
  ballRs.io.commit_i.ball2.valid := bbus.io.cmdResp.valid
  ballRs.io.commit_i.ball2.bits := bbus.io.cmdResp.bits
  ballRs.io.commit_i.ball3.valid := bbus.io.cmdResp.valid
  ballRs.io.commit_i.ball3.bits := bbus.io.cmdResp.bits
  ballRs.io.commit_i.ball4.valid := bbus.io.cmdResp.valid
  ballRs.io.commit_i.ball4.bits := bbus.io.cmdResp.bits

  // ready信号汇聚
  bbus.io.cmdResp.ready := ballRs.io.commit_i.ball1.ready ||
                           ballRs.io.commit_i.ball2.ready ||
                           ballRs.io.commit_i.ball3.ready ||
                           ballRs.io.commit_i.ball4.ready

//---------------------------------------------------------------------------
// BBus -> Mem Domain
//---------------------------------------------------------------------------
  bbus.io.sramRead  <> io.sramRead
  bbus.io.sramWrite <> io.sramWrite
  bbus.io.accRead   <> io.accRead
  bbus.io.accWrite  <> io.accWrite

//---------------------------------------------------------------------------
// BallController -> RoCC
//---------------------------------------------------------------------------
  io.roccResp <> ballRs.io.rs_rocc_o.resp
  io.busy := ballRs.io.rs_rocc_o.busy

  override lazy val desiredName = "BallDomain"
}
