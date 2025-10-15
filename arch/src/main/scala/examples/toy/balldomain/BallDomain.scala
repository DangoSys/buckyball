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
import examples.toy.balldomain.rs.BallRSModule
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
  val ballRs = Module(new BallRSModule)
  ballRs.io.ball_decode_cmd_i <> ballDecoder.io.ball_decode_cmd_o

//---------------------------------------------------------------------------
// BallReservationStation -> BBus
//---------------------------------------------------------------------------
  bbus.io.cmdReq <> ballRs.io.issue_o.balls
  ballRs.io.commit_i.balls <> bbus.io.cmdResp

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
