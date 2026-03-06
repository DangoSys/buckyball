package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig
import framework.balldomain.prototype.systolicarray.configs.SystolicBallParam

@instantiable
class SystolicArrayLoad(val b: GlobalConfig) extends Module {
  val config      = SystolicBallParam()
  val InputNum    = 16
  val bankWidth   = b.memDomain.bankWidth
  val inputWidth  = config.inputWidth
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  val inBW        = ballMapping.inBW

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))
    val ctrl_ld_i    = Flipped(Decoupled(new ctrl_ld_req(b)))
    val ld_ex_o      = Decoupled(new ld_ex_req(b))
    val op1_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val op2_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
  })

  val op1_bank            = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2_bank            = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op1_addr            = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val op2_addr            = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val iter                = RegInit(0.U(10.W))
  val op1_iter_counter    = RegInit(0.U(10.W))
  val op2_iter_counter    = RegInit(0.U(10.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)
  val ld_ex_iter_reg      = RegInit(0.U(10.W))

  val bankRespQueue0 = Module(new Queue(new SramReadResp(b), entries = 8))
  val bankRespQueue1 = Module(new Queue(new SramReadResp(b), entries = 8))

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
  }

  io.op1_bank_o      := op1_bank
  io.op2_bank_o      := op2_bank
  io.ctrl_ld_i.ready := state === idle

  bankRespQueue0.io.enq <> io.bankReadResp(0)
  bankRespQueue1.io.enq <> io.bankReadResp(1)

  when(io.ctrl_ld_i.fire) {
    op1_bank         := io.ctrl_ld_i.bits.op1_bank
    op2_bank         := io.ctrl_ld_i.bits.op2_bank
    op1_addr         := io.ctrl_ld_i.bits.op1_bank_addr
    op2_addr         := io.ctrl_ld_i.bits.op2_bank_addr
    iter             := io.ctrl_ld_i.bits.iter
    op1_iter_counter := 0.U
    op2_iter_counter := 0.U
    state            := busy
  }

  when(state === busy && io.ld_ex_o.ready) {
    io.bankReadReq(0).valid     := op1_iter_counter < iter
    io.bankReadReq(0).bits.addr := op1_addr + op1_iter_counter
    op1_iter_counter            := Mux(io.bankReadReq(0).ready, op1_iter_counter + 1.U, op1_iter_counter)
  }

  when(state === busy && io.ld_ex_o.ready) {
    io.bankReadReq(1).valid     := op2_iter_counter < iter
    io.bankReadReq(1).bits.addr := op2_addr + op2_iter_counter
    op2_iter_counter            := Mux(io.bankReadReq(1).ready, op2_iter_counter + 1.U, op2_iter_counter)
  }

// -----------------------------------------------------------------------------
// SRAM returns data and passes to EX unit
// -----------------------------------------------------------------------------
  val both_valid = bankRespQueue0.io.deq.valid && bankRespQueue1.io.deq.valid

  io.ld_ex_o.valid := both_valid
  when(both_valid) {
    io.ld_ex_o.bits.op1  := bankRespQueue0.io.deq.bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
    io.ld_ex_o.bits.op2  := bankRespQueue1.io.deq.bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
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
