package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp, SramWriteIO}

class ctrl_ld_req(b: GlobalConfig) extends Bundle {
  val op1_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val iter          = UInt(b.frontend.iter_len.W)
}

@instantiable
class SystolicArrayUnit(val b: GlobalConfig) extends Module {
  val InputNum   = 16
  val inputWidth = 8
  val accWidth   = 32

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  val ctrl:  Instance[SystolicArrayCtrl]  = Instantiate(new SystolicArrayCtrl(b))
  val load:  Instance[SystolicArrayLoad]  = Instantiate(new SystolicArrayLoad(b))
  val ex:    Instance[SystolicArrayEX]    = Instantiate(new SystolicArrayEX(b))
  val store: Instance[SystolicArrayStore] = Instantiate(new SystolicArrayStore(b))

  ctrl.io.cmdReq <> io.cmdReq
  io.cmdResp <> ctrl.io.cmdResp_o

  ctrl.io.ctrl_ld_o <> load.io.ctrl_ld_i
  ctrl.io.ctrl_ex_o <> ex.io.ctrl_ex_i
  ctrl.io.ctrl_st_o <> store.io.ctrl_st_i

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req <> load.io.bankReadReq(i)
    load.io.bankReadResp(i) <> io.bankRead(i).io.resp
    if (i == 0) {
      io.bankRead(i).bank_id  := load.io.op1_bank_o
      io.bankRead(i).group_id := 0.U
    } else if (i == 1) {
      io.bankRead(i).bank_id  := load.io.op2_bank_o
      io.bankRead(i).group_id := 0.U
    }
  }

  load.io.ld_ex_o <> ex.io.ld_ex_i
  ex.io.ex_st_o <> store.io.ex_st_i
  ctrl.io.cmdResp_i <> store.io.cmdResp_o

  for (i <- 0 until outBW) {
    io.bankWrite(i).io <> store.io.bankWrite(i)
    io.bankWrite(i).bank_id           := store.io.wr_bank_o
    io.bankWrite(i).io.req.bits.wmode := true.B
    io.bankWrite(i).group_id          := i.U
  }

  val hasInput  = RegInit(false.B)
  val hasOutput = RegInit(false.B)

  when(io.cmdReq.fire) {
    hasInput := true.B
  }
  when(io.cmdResp.fire) {
    hasOutput := false.B
    hasInput  := false.B
  }
  when(io.cmdResp.valid && !hasOutput) {
    hasOutput := true.B
  }

  io.status.idle    := !hasInput && !hasOutput
  io.status.running := hasOutput
}
