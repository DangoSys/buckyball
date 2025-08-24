package examples.toy.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.PostGDCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.rocket.RoCCResponseBB

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
}

// Ball Domain 顶层
class BallDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new BallDomainIO)
  
//---------------------------------------------------------------------------
// Decoder -> BallReservationStation
//---------------------------------------------------------------------------
  val ballDecoder = Module(new BallDomainDecoder)
  ballDecoder.io.raw_cmd_i <> io.gDecoderIn
  
//---------------------------------------------------------------------------
// Decoder -> BallReservationStation
//---------------------------------------------------------------------------
  val ballRs = Module(new BallReservationStation)
  ballRs.io.ball_decode_cmd_i <> ballDecoder.io.ball_decode_cmd_o

//---------------------------------------------------------------------------
// BallReservationStation -> ExecuteController
//---------------------------------------------------------------------------
  val ballController = Module(new ExecuteController)

  ballController.io.cmdReq <> ballRs.io.issue_o
  ballRs.io.commit_i <> ballController.io.cmdResp
  
//---------------------------------------------------------------------------
// ExecuteController -> Mem Domain
//---------------------------------------------------------------------------
  ballController.io.sramRead  <> io.sramRead
  ballController.io.sramWrite <> io.sramWrite
  ballController.io.accRead   <> io.accRead
  ballController.io.accWrite  <> io.accWrite
  
//---------------------------------------------------------------------------
// ExecuteController -> RoCC
//---------------------------------------------------------------------------
  io.roccResp <> ballRs.io.rs_rocc_o.resp
  io.busy := ballRs.io.rs_rocc_o.busy
}
