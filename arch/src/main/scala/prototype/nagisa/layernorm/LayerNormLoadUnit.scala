package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO}

class LayerNormLoadUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_ld_i = Flipped(Decoupled(new LNCtrlLdReq))

    // Memory read interfaces
    val sramReadReq  = Vec(b.sp_banks, Decoupled(new SramReadReq(b.spad_bank_entries)))
    val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(b.spad_w))))
    val accReadReq   = Vec(b.acc_banks, Decoupled(new SramReadReq(b.acc_bank_entries)))
    val accReadResp  = Vec(b.acc_banks, Flipped(Decoupled(new SramReadResp(b.acc_w))))

    // Data outputs
    val ld_reduce_o = Decoupled(new LNLdReduceReq)
    val ld_norm_param_o = Valid(new LNLdNormParam)
  })

  // State machine
  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

  // Control registers
  val op1_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val op1_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter_reg      = RegInit(0.U(10.W))
  val is_acc_reg    = RegInit(false.B)
  val norm_dim_reg  = RegInit(0.U(12.W))
  val use_affine_reg = RegInit(false.B)

  // Request and response counters
  val req_cnt  = RegInit(0.U(20.W))
  val resp_cnt = RegInit(0.U(20.W))

  // Output buffer registers (KEY ADDITION - like GELU)
  val ld_reduce_valid_reg = RegInit(false.B)
  val ld_reduce_data_reg = Reg(Vec(b.veclane, SInt(32.W)))
  val ld_reduce_batch_idx_reg = RegInit(0.U(10.W))
  val ld_reduce_vec_idx_reg = RegInit(0.U(12.W))
  val ld_reduce_is_last_reg = RegInit(false.B)

  // Accept control signals
  io.ctrl_ld_i.ready := state === idle
  when(io.ctrl_ld_i.fire) {
    op1_bank       := io.ctrl_ld_i.bits.op1_bank
    op1_bank_addr  := io.ctrl_ld_i.bits.op1_bank_addr
    iter_reg       := io.ctrl_ld_i.bits.iter
    is_acc_reg     := io.ctrl_ld_i.bits.is_acc
    norm_dim_reg   := io.ctrl_ld_i.bits.norm_dim
    use_affine_reg := io.ctrl_ld_i.bits.use_affine

    req_cnt  := 0.U
    resp_cnt := 0.U
    state    := busy
  }

  // Calculate total number of vectors to load
  val total_vecs = iter_reg * norm_dim_reg

  // Default outputs for unused interfaces
  for (i <- 0 until b.sp_banks) {
    io.sramReadReq(i).valid := false.B
    io.sramReadReq(i).bits.addr := 0.U
    io.sramReadReq(i).bits.fromDMA := false.B
  }
  for (i <- 0 until b.acc_banks) {
    io.accReadReq(i).valid := false.B
    io.accReadReq(i).bits.addr := 0.U
    io.accReadReq(i).bits.fromDMA := false.B
  }
  io.ld_norm_param_o.valid := false.B
  io.ld_norm_param_o.bits.gamma := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_norm_param_o.bits.beta := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_norm_param_o.bits.vec_idx := 0.U

  // Send read requests (only when buffer is ready)
  when(state === busy && req_cnt < total_vecs && (!ld_reduce_valid_reg || io.ld_reduce_o.ready)) {
    val addr = op1_bank_addr + req_cnt
    when(is_acc_reg) {
      val bank_id = op1_bank
      when(bank_id < b.acc_banks.U) {
        io.accReadReq(bank_id).valid := true.B
        io.accReadReq(bank_id).bits.addr := addr
        io.accReadReq(bank_id).bits.fromDMA := false.B
        when(io.accReadReq(bank_id).ready) {
          req_cnt := req_cnt + 1.U
        }
      }
    }.otherwise {
      val bank_id = op1_bank
      when(bank_id < b.sp_banks.U) {
        io.sramReadReq(bank_id).valid := true.B
        io.sramReadReq(bank_id).bits.addr := addr
        io.sramReadReq(bank_id).bits.fromDMA := false.B
        when(io.sramReadReq(bank_id).ready) {
          req_cnt := req_cnt + 1.U
        }
      }
    }
  }

  // Handle read responses - set ready signal
  io.sramReadResp.foreach { resp => resp.ready := !ld_reduce_valid_reg || io.ld_reduce_o.ready }
  io.accReadResp.foreach { resp => resp.ready := !ld_reduce_valid_reg || io.ld_reduce_o.ready }

  // Process read responses
  when(is_acc_reg) {
    val bank_id = op1_bank
    when(io.accReadResp(bank_id).valid && (!ld_reduce_valid_reg || io.ld_reduce_o.ready)) {
      val raw_data = io.accReadResp(bank_id).bits.data

      // Convert to INT32 vectors
      val acc_vec = raw_data.asTypeOf(Vec(b.veclane, SInt(32.W)))

      // Calculate batch and vector indices
      val vec_idx = resp_cnt % norm_dim_reg
      val batch_idx = resp_cnt / norm_dim_reg
      val is_last = (vec_idx === norm_dim_reg - 1.U)

      // Store in buffer registers
      ld_reduce_valid_reg := true.B
      ld_reduce_data_reg := acc_vec
      ld_reduce_batch_idx_reg := batch_idx
      ld_reduce_vec_idx_reg := vec_idx
      ld_reduce_is_last_reg := is_last
      resp_cnt := resp_cnt + 1.U
    }.elsewhen(io.ld_reduce_o.ready) {
      ld_reduce_valid_reg := false.B
    }
  }.otherwise {
    val bank_id = op1_bank
    when(io.sramReadResp(bank_id).valid && (!ld_reduce_valid_reg || io.ld_reduce_o.ready)) {
      val raw_data = io.sramReadResp(bank_id).bits.data

      // Convert INT8 to INT32 with sign extension
      val sram_vec = raw_data.asTypeOf(Vec(b.veclane, SInt(8.W)))
      val data_vec = Wire(Vec(b.veclane, SInt(32.W)))
      for (i <- 0 until b.veclane) {
        data_vec(i) := sram_vec(i)  // Auto sign-extend
      }

      // Calculate batch and vector indices
      val vec_idx = resp_cnt % norm_dim_reg
      val batch_idx = resp_cnt / norm_dim_reg
      val is_last = (vec_idx === norm_dim_reg - 1.U)

      // Store in buffer registers
      ld_reduce_valid_reg := true.B
      ld_reduce_data_reg := data_vec
      ld_reduce_batch_idx_reg := batch_idx
      ld_reduce_vec_idx_reg := vec_idx
      ld_reduce_is_last_reg := is_last
      resp_cnt := resp_cnt + 1.U
    }.elsewhen(io.ld_reduce_o.ready) {
      ld_reduce_valid_reg := false.B
    }
  }

  // Output from buffer registers
  io.ld_reduce_o.valid := ld_reduce_valid_reg
  io.ld_reduce_o.bits.data := ld_reduce_data_reg
  io.ld_reduce_o.bits.batch_idx := ld_reduce_batch_idx_reg
  io.ld_reduce_o.bits.vec_idx := ld_reduce_vec_idx_reg
  io.ld_reduce_o.bits.is_last := ld_reduce_is_last_reg

  // Return to idle when all responses received and buffer cleared
  when(state === busy && resp_cnt === total_vecs && !ld_reduce_valid_reg) {
    state := idle
  }
}
