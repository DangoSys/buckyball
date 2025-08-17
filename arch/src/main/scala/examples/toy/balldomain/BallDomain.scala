package examples.toy.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.PostDecodeCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig

// Ball Domain的输入输出接口
class BallDomainIO(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // 来自GlobalDecoder的命令分发接口
  val globalDecoderIn = Flipped(Decoupled(new PostDecodeCmd))
  
  // 连接到Scratchpad的执行接口  
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  
  // RoCC响应接口
  val roccResp = Decoupled(new RoCCResponse()(p))
  val busy = Output(Bool())
}

// Ball Domain 顶层
class BallDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new BallDomainIO)
  
//---------------------------------------------------------------------------
// Decoder -> ExReservationStation
//---------------------------------------------------------------------------
  val ballDecoder = Module(new ExDomainDecoder)
  ballDecoder.io.post_decode_cmd_i <> io.globalDecoderIn
  io.globalDecoderIn.ready := ballDecoder.io.post_decode_cmd_i.ready
  
//---------------------------------------------------------------------------
// Decoder -> ExReservationStation
//---------------------------------------------------------------------------
  val ballReservationStation = Module(new ExReservationStation)
  ballReservationStation.io.ex_decode_cmd_i <> ballDecoder.io.ex_decode_cmd_o
  io.globalDecoderIn.ready := ballDecoder.io.post_decode_cmd_i.ready

//---------------------------------------------------------------------------
// ExReservationStation -> ExecuteController
//---------------------------------------------------------------------------
  val ballController = Module(new ExecuteController)

  ballController.io.cmdReq <> ballReservationStation.io.issue_o
  ballReservationStation.io.commit_i <> ballController.io.cmdResp
  
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
  io.roccResp <> ballReservationStation.io.rs_rocc_o.resp
  io.busy := ballReservationStation.io.rs_rocc_o.busy
}
