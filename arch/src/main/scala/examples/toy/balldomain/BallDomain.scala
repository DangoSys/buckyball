package examples.toy.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy.{LazyModule, LazyModuleImp}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.PostGDCmd
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.toy.balldomain.rs.BallRSModule
import examples.toy.balldomain.bbus.BBusModule

// Ball Domain input/output interface
class BallDomainIO(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  // Issue interface from global RS (single channel)
  val global_issue_i = Flipped(Decoupled(new framework.builtin.frontend.globalrs.GlobalRsIssue))

  // Report completion to global RS (single channel)
  val global_complete_o = Decoupled(new framework.builtin.frontend.globalrs.GlobalRsComplete)

  // Execution interface connected to Scratchpad
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
}

// Ball Domain top level - uses new simplified BBus architecture
class BallDomain(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {

  val io = IO(new BallDomainIO)

  // Create new BBus module
  val bbus = Module(new BBusModule)

//---------------------------------------------------------------------------
// Global RS -> Decoder (receive global issue and construct PostGDCmd)
//---------------------------------------------------------------------------
  val ballDecoder = Module(new BallDomainDecoder)

  // Convert global RS issue to Decoder input format
  ballDecoder.io.raw_cmd_i.valid := io.global_issue_i.valid
  ballDecoder.io.raw_cmd_i.bits  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready := ballDecoder.io.raw_cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> Local BallRS (multi-channel issue to each Ball device)
//---------------------------------------------------------------------------
  val ballRs = Module(new BallRSModule)

  // Connect decoded instruction and global rob_id
  ballRs.io.ball_decode_cmd_i.valid := ballDecoder.io.ball_decode_cmd_o.valid
  ballRs.io.ball_decode_cmd_i.bits.cmd := ballDecoder.io.ball_decode_cmd_o.bits
  ballRs.io.ball_decode_cmd_i.bits.rob_id := io.global_issue_i.bits.rob_id
  ballDecoder.io.ball_decode_cmd_o.ready := ballRs.io.ball_decode_cmd_i.ready

//---------------------------------------------------------------------------
// Local BallRS -> BBus (multi-channel)
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
// Local RS completion signal -> Global RS (single channel, includes global rob_id)
//---------------------------------------------------------------------------
  io.global_complete_o <> ballRs.io.complete_o

  override lazy val desiredName = "BallDomain"
}
