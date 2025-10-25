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
  // 来自全局RS的发射接口 (单通道)
  val global_issue_i = Flipped(Decoupled(new framework.builtin.frontend.globalrs.GlobalRsIssue))

  // 向全局RS报告完成 (单通道)
  val global_complete_o = Decoupled(new framework.builtin.frontend.globalrs.GlobalRsComplete)

  // 连接到Scratchpad的执行接口
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
}

// Ball Domain 顶层 - 使用新的简化BBus架构
class BallDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {

  val io = IO(new BallDomainIO)

  // 创建新的BBus模块
  val bbus = Module(new BBusModule)

//---------------------------------------------------------------------------
// 全局RS -> Decoder (接收全局发射并构造PostGDCmd)
//---------------------------------------------------------------------------
  val ballDecoder = Module(new BallDomainDecoder)

  // 将全局RS的发射转换为Decoder的输入格式
  ballDecoder.io.raw_cmd_i.valid := io.global_issue_i.valid
  ballDecoder.io.raw_cmd_i.bits  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready := ballDecoder.io.raw_cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> 局部BallRS (多通道发射到各Ball设备)
//---------------------------------------------------------------------------
  val ballRs = Module(new BallRSModule)

  // 连接解码后的指令和全局rob_id
  ballRs.io.ball_decode_cmd_i.valid := ballDecoder.io.ball_decode_cmd_o.valid
  ballRs.io.ball_decode_cmd_i.bits.cmd := ballDecoder.io.ball_decode_cmd_o.bits
  ballRs.io.ball_decode_cmd_i.bits.rob_id := io.global_issue_i.bits.rob_id
  ballDecoder.io.ball_decode_cmd_o.ready := ballRs.io.ball_decode_cmd_i.ready

//---------------------------------------------------------------------------
// 局部BallRS -> BBus (多通道)
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
// 局部RS完成信号 -> 全局RS (单通道，包含全局rob_id)
//---------------------------------------------------------------------------
  io.global_complete_o <> ballRs.io.complete_o

  override lazy val desiredName = "BallDomain"
}
