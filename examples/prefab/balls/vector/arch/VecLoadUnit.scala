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
import framework.ip.memory.bank.{SramReadReq, SramReadResp}
import examples.balls.vector.configs.VectorBallParam

class ctrl_ld_req(frontendParam: FrontendParam, memParam: MemDomainParam, val ballParam: VectorBallParam)
    extends Bundle {
  val op1_bank      = UInt(frontendParam.bank_id_len.W)
  val op1_bank_addr = UInt(log2Up(memParam.bankEntries).W)
  val op2_bank      = UInt(frontendParam.bank_id_len.W)
  val op2_bank_addr = UInt(log2Up(memParam.bankEntries).W)
  val iter          = UInt(frontendParam.iter_len.W)
  val mode          = UInt(1.W)
}

@instantiable
class VecLoadUnit(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     VectorBallParam)
    extends Module {
  val config       = ballParam
  val InputNum     = config.lane
  val inputWidth   = config.inputWidth
  val bankWidth    = memParam.bankWidth
  val rob_id_width = log2Up(frontendParam.rob_entries)

  // Use this instance's configured bandwidth.
  val ballMapping = instance

  val inBW = ballMapping.inBW

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(memParam)))
    val bankReadResp =
      Vec(inBW, Flipped(Decoupled(new SramReadResp(memParam))))
    val ctrl_ld_i    = Flipped(Decoupled(new ctrl_ld_req(frontendParam, memParam, ballParam)))
    val ld_ex_o      = Decoupled(new ld_ex_req(frontendParam, ballParam))
    val op1_bank_o   = Output(UInt(frontendParam.bank_id_len.W))
    val op2_bank_o   = Output(UInt(frontendParam.bank_id_len.W))
  })

  val op1_bank            = RegInit(0.U(frontendParam.bank_id_len.W))
  val op2_bank            = RegInit(0.U(frontendParam.bank_id_len.W))
  val op1_addr            = RegInit(0.U(log2Up(memParam.bankEntries).W))
  val op2_addr            = RegInit(0.U(log2Up(memParam.bankEntries).W))
  val iter                = RegInit(0.U(frontendParam.iter_len.W))
  val op1_iter_counter    = RegInit(0.U(frontendParam.iter_len.W))
  val op2_iter_counter    = RegInit(0.U(frontendParam.iter_len.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)
  val ld_ex_op1_reg       = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_op2_reg       = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_iter_reg      = RegInit(0.U(frontendParam.iter_len.W))
  val wait1_reg           = RegInit(false.B)
  val wait2_reg           = RegInit(false.B)
  val wait1_cnt           = RegInit(0.U(7.W))
  val wait2_cnt           = RegInit(0.U(7.W))

  val bankRespQueue0 = Instantiate(new Queue(
    new SramReadResp(memParam),
    8,
    false,
    false
  ))

  val bankRespQueue1 = Instantiate(new Queue(
    new SramReadResp(memParam),
    8,
    false,
    false
  ))

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
  }

  io.op1_bank_o      := op1_bank
  io.op2_bank_o      := op2_bank
  io.ctrl_ld_i.ready := state === idle

  bankRespQueue0.io.enq <> io.bankReadResp(0)
  bankRespQueue1.io.enq <> io.bankReadResp(1)

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  when(io.ctrl_ld_i.fire) {
    op1_bank         := io.ctrl_ld_i.bits.op1_bank
    op2_bank         := io.ctrl_ld_i.bits.op2_bank
    op1_addr         := io.ctrl_ld_i.bits.op1_bank_addr
    op2_addr         := io.ctrl_ld_i.bits.op2_bank_addr
    iter             := io.ctrl_ld_i.bits.iter
    op1_iter_counter := 0.U
    op2_iter_counter := 0.U
    state            := busy
    assert(io.ctrl_ld_i.bits.iter > 0.U, "iter should be greater than 0")
  }

// -----------------------------------------------------------------------------
// Send SRAM read request
// -----------------------------------------------------------------------------
  //wait1_reg := Mux(io.run , 0.U, wait1_reg)

  //wait2_reg := Mux(io.run , 0.U, wait2_reg)

  when(state === busy && io.ld_ex_o.ready && !wait1_reg) {
    io.bankReadReq(0).valid     := op1_iter_counter < iter
    io.bankReadReq(0).bits.addr := op1_addr + op1_iter_counter
    op1_iter_counter            := Mux(
      io.bankReadReq(0).ready,
      op1_iter_counter + 1.U,
      op1_iter_counter
    )
    wait1_reg                   := Mux((op1_iter_counter + 1.U) % 16.U === 0.U, 1.U, 0.U)
    when(io.bankReadReq(0).fire) {}
  }

  when(state === busy && io.ld_ex_o.ready && !wait2_reg) {
    io.bankReadReq(1).valid     := op2_iter_counter < iter
    io.bankReadReq(1).bits.addr := op2_addr + op2_iter_counter
    op2_iter_counter            := Mux(
      io.bankReadReq(1).ready,
      op2_iter_counter + 1.U,
      op2_iter_counter
    )
    wait2_reg                   := Mux((op2_iter_counter + 1.U) % 16.U === 0.U, 1.U, 0.U)
    when(io.bankReadReq(1).fire) {}
  }

  when(wait1_reg) {
    wait1_cnt := wait1_cnt + 1.U
    when(wait1_cnt === 32.U) {
      wait1_reg := false.B
      wait1_cnt := 0.U
    }
  }

  when(wait2_reg) {
    wait2_cnt := wait2_cnt + 1.U
    when(wait2_cnt === 32.U) {
      wait2_reg := false.B
      wait2_cnt := 0.U
    }
  }

// -----------------------------------------------------------------------------
// SRAM returns data and passes to EX unit
// -----------------------------------------------------------------------------
  val both_valid = bankRespQueue0.io.deq.valid && bankRespQueue1.io.deq.valid

  io.ld_ex_o.valid := both_valid
  when(both_valid) {
    io.ld_ex_o.bits.op1  := bankRespQueue0.io.deq.bits.data
      .asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
    io.ld_ex_o.bits.op2  := bankRespQueue1.io.deq.bits.data
      .asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
    io.ld_ex_o.bits.iter := ld_ex_iter_reg
  }.otherwise {
    io.ld_ex_o.bits.iter := 0.U
    io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
  }

  // Only dequeue and advance iter counter on successful handshake
  bankRespQueue0.io.deq.ready := io.ld_ex_o.fire
  bankRespQueue1.io.deq.ready := io.ld_ex_o.fire

  when(io.ld_ex_o.fire) {
    ld_ex_iter_reg := ld_ex_iter_reg + 1.U
  }

// -----------------------------------------------------------------------------
// Reset op1_iter_counter and return to idle state
// -----------------------------------------------------------------------------

  when(state === busy && ld_ex_iter_reg === iter) {
    state            := idle
    op1_iter_counter := 0.U
    op2_iter_counter := 0.U
    ld_ex_iter_reg   := 0.U
  }

}
