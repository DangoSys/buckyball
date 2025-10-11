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
  val idle :: load_data :: load_params :: done :: Nil = Enum(4)
  val state = RegInit(idle)

  // Control registers
  val op1_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val op1_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter_reg      = RegInit(0.U(10.W))
  val is_acc_reg    = RegInit(false.B)
  val norm_dim_reg  = RegInit(0.U(12.W))
  val param_bank_reg = RegInit(0.U(2.W))
  val gamma_addr_reg = RegInit(0.U(12.W))
  val beta_addr_reg  = RegInit(0.U(12.W))
  val use_affine_reg = RegInit(false.B)

  // Loop counters
  val batch_cnt = RegInit(0.U(10.W))
  val vec_cnt   = RegInit(0.U(12.W))

  // Accept control signals
  io.ctrl_ld_i.ready := state === idle
  when(io.ctrl_ld_i.fire) {
    op1_bank       := io.ctrl_ld_i.bits.op1_bank
    op1_bank_addr  := io.ctrl_ld_i.bits.op1_bank_addr
    iter_reg       := io.ctrl_ld_i.bits.iter
    is_acc_reg     := io.ctrl_ld_i.bits.is_acc
    norm_dim_reg   := io.ctrl_ld_i.bits.norm_dim
    param_bank_reg := io.ctrl_ld_i.bits.param_bank
    gamma_addr_reg := io.ctrl_ld_i.bits.gamma_addr
    beta_addr_reg  := io.ctrl_ld_i.bits.beta_addr
    use_affine_reg := io.ctrl_ld_i.bits.use_affine

    batch_cnt := 0.U
    vec_cnt   := 0.U
    state     := load_data
  }

  // Default outputs
  for (i <- 0 until b.sp_banks) {
    io.sramReadReq(i).valid := false.B
    io.sramReadReq(i).bits.addr := 0.U
    io.sramReadReq(i).bits.fromDMA := false.B
    io.sramReadResp(i).ready := false.B
  }
  for (i <- 0 until b.acc_banks) {
    io.accReadReq(i).valid := false.B
    io.accReadReq(i).bits.addr := 0.U
    io.accReadReq(i).bits.fromDMA := false.B
    io.accReadResp(i).ready := false.B
  }
  io.ld_reduce_o.valid := false.B
  io.ld_reduce_o.bits.data := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_reduce_o.bits.batch_idx := 0.U
  io.ld_reduce_o.bits.vec_idx := 0.U
  io.ld_reduce_o.bits.is_last := false.B

  io.ld_norm_param_o.valid := false.B
  io.ld_norm_param_o.bits.gamma := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_norm_param_o.bits.beta := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_norm_param_o.bits.vec_idx := 0.U

  // State machine logic
  switch(state) {
    is(idle) {
      // Wait for control signal
    }

    is(load_data) {
      // Calculate address
      val addr = op1_bank_addr + batch_cnt * norm_dim_reg + vec_cnt

      // Issue read request
      when(is_acc_reg) {
        val bank_id = op1_bank
        when(bank_id < b.acc_banks.U) {
          io.accReadReq(bank_id).valid := true.B
          io.accReadReq(bank_id).bits.addr := addr
          io.accReadReq(bank_id).bits.fromDMA := false.B
        }
      }.otherwise {
        val bank_id = op1_bank
        when(bank_id < b.sp_banks.U) {
          io.sramReadReq(bank_id).valid := true.B
          io.sramReadReq(bank_id).bits.addr := addr
          io.sramReadReq(bank_id).bits.fromDMA := false.B
        }
      }

      // Wait for response
      val resp_valid = Mux(is_acc_reg,
        io.accReadResp(op1_bank).valid,
        io.sramReadResp(op1_bank).valid
      )

      when(resp_valid) {
        val raw_data = Mux(is_acc_reg,
          io.accReadResp(op1_bank).bits.data,
          io.sramReadResp(op1_bank).bits.data
        )

        // Mark as ready
        when(is_acc_reg) {
          io.accReadResp(op1_bank).ready := true.B
        }.otherwise {
          io.sramReadResp(op1_bank).ready := true.B
        }

        // Convert to INT32 vectors
        val data_vec = Wire(Vec(b.veclane, SInt(32.W)))
        when(is_acc_reg) {
          // ACC mode: 512-bit data, 16xINT32
          val acc_vec = raw_data.asTypeOf(Vec(b.veclane, SInt(32.W)))
          data_vec := acc_vec
        }.otherwise {
          // SRAM mode: 128-bit data, 16xINT8, sign-extend to INT32
          val sram_vec = raw_data.asTypeOf(Vec(b.veclane, SInt(8.W)))
          for (i <- 0 until b.veclane) {
            data_vec(i) := sram_vec(i)  // Auto sign-extend from INT8 to INT32
          }
        }

        // Send to reduce unit
        io.ld_reduce_o.valid := true.B
        io.ld_reduce_o.bits.data := data_vec
        io.ld_reduce_o.bits.batch_idx := batch_cnt
        io.ld_reduce_o.bits.vec_idx := vec_cnt
        io.ld_reduce_o.bits.is_last := (vec_cnt === norm_dim_reg - 1.U)

        when(io.ld_reduce_o.ready) {
          // Move to next vector
          when(vec_cnt === norm_dim_reg - 1.U) {
            vec_cnt := 0.U
            // Move to next batch
            when(batch_cnt === iter_reg - 1.U) {
              // All batches loaded
              when(use_affine_reg) {
                state := load_params
                vec_cnt := 0.U
              }.otherwise {
                state := done
              }
            }.otherwise {
              batch_cnt := batch_cnt + 1.U
            }
          }.otherwise {
            vec_cnt := vec_cnt + 1.U
          }
        }
      }
    }

    is(load_params) {
      // Load gamma and beta parameters
      // This is a simplified version; a full implementation would load parameters
      // For now, we just transition to done
      state := done
    }

    is(done) {
      // Stay in done state
    }
  }
}
