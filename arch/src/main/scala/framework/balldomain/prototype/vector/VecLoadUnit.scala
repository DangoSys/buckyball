package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig
import framework.balldomain.prototype.vector.configs.VectorBallParam

class ctrl_ld_req(b: GlobalConfig) extends Bundle {
  val op1_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val iter          = UInt(10.W)
  val mode          = UInt(1.W)
}

@instantiable
class VecLoadUnit(val b: GlobalConfig) extends Module {
  val config       = VectorBallParam()
  val InputNum     = config.lane
  val inputWidth   = config.inputWidth
  val bankWidth    = b.memDomain.bankWidth
  val rob_id_width = log2Up(b.frontend.rob_entries)

  // Get bandwidth from config (use first VecBall mapping)
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
    .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
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
  val iter_counter        = RegInit(0.U(10.W))
  val mode                = RegInit(0.U(1.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)
  val ld_ex_valid_reg     = RegInit(false.B)
  val ld_ex_op1_reg       = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_op2_reg       = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_iter_reg      = RegInit(0.U(10.W))

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
  }

  io.op1_bank_o      := op1_bank
  io.op2_bank_o      := op2_bank
  io.ctrl_ld_i.ready := state === idle

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------

  when(io.ctrl_ld_i.fire) {
    op1_bank     := io.ctrl_ld_i.bits.op1_bank
    op2_bank     := io.ctrl_ld_i.bits.op2_bank
    op1_addr     := io.ctrl_ld_i.bits.op1_bank_addr
    op2_addr     := io.ctrl_ld_i.bits.op2_bank_addr
    iter         := io.ctrl_ld_i.bits.iter
    iter_counter := 0.U
    state        := busy
    mode         := io.ctrl_ld_i.bits.mode
    assert(io.ctrl_ld_i.bits.iter > 0.U, "iter should be greater than 0")
  }
  io.bankReadResp.foreach { resp =>
    resp.ready := state === busy
  }
  when(mode === 0.U) {
// -----------------------------------------------------------------------------
// Send SRAM read request
// -----------------------------------------------------------------------------
    when(state === busy && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      io.bankReadReq(0).valid     := iter_counter < iter
      io.bankReadReq(0).bits.addr := op1_addr + iter_counter
      io.bankReadReq(1).valid     := iter_counter < iter
      io.bankReadReq(1).bits.addr := op2_addr + iter_counter
      iter_counter                := Mux(io.bankReadReq(0).ready && io.bankReadReq(1).ready, iter_counter + 1.U, iter_counter)
    }

// -----------------------------------------------------------------------------
// SRAM returns data and passes to EX unit
// -----------------------------------------------------------------------------
    when(io.bankReadResp(0).valid &&
      (!ld_ex_valid_reg || io.ld_ex_o.ready) && (state === busy)) {
      ld_ex_valid_reg := true.B
      ld_ex_op1_reg   := io.bankReadResp(0).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      ld_ex_op2_reg   := io.bankReadResp(1).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      ld_ex_iter_reg  := iter_counter
    }.elsewhen(io.ld_ex_o.ready) {
      ld_ex_valid_reg := false.B
    }

    io.ld_ex_o.valid     := ld_ex_valid_reg
    io.ld_ex_o.bits.op1  := ld_ex_op1_reg
    io.ld_ex_o.bits.op2  := ld_ex_op2_reg
    io.ld_ex_o.bits.iter := ld_ex_iter_reg

// -----------------------------------------------------------------------------
// Reset iter_counter and return to idle state
// -----------------------------------------------------------------------------

    when(state === busy && iter_counter === iter && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      state        := idle
      iter_counter := 0.U
    }
  }.otherwise {
    io.ld_ex_o.valid     := false.B
    io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    io.ld_ex_o.bits.iter := 0.U
    when(state === busy && io.bankReadResp(0).valid) {
      iter_counter                := iter_counter + 1.U
      ld_ex_op1_reg               := io.bankReadResp(0).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.bankReadReq(1).valid     := true.B
      io.bankReadReq(1).bits.addr := op2_addr + iter_counter
    }
    when(state === busy && io.bankReadResp(1).valid && RegNext(io.bankReadResp(0).valid)) {
      io.ld_ex_o.valid     := true.B
      io.ld_ex_o.bits.op1  := ld_ex_op1_reg
      io.ld_ex_o.bits.op2  := io.bankReadResp(1).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.iter := iter_counter - 1.U
    }
    when(state === busy && iter_counter === iter) {
      state        := idle
      iter_counter := 0.U
    }
  }

}
