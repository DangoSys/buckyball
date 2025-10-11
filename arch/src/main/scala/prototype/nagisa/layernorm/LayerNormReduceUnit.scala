package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class LayerNormReduceUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_reduce_i = Flipped(Decoupled(new LNCtrlReduceReq))
    val ld_reduce_i = Flipped(Decoupled(new LNLdReduceReq))
    val reduce_norm_o = Decoupled(new LNReduceNormReq)
  })

  // State machine
  val idle :: accumulate :: compute :: done :: Nil = Enum(4)
  val state = RegInit(idle)

  // Control registers
  val iter_reg = RegInit(0.U(10.W))
  val norm_dim_reg = RegInit(0.U(12.W))

  // Accumulators
  val sum_acc = RegInit(0.S(48.W))  // Sum of all elements
  val sum_sq_acc = RegInit(0.S(64.W))  // Sum of squared elements
  val vec_cnt = RegInit(0.U(12.W))
  val batch_cnt = RegInit(0.U(10.W))
  val cur_batch = RegInit(0.U(10.W))

  // Computed statistics
  val mean = RegInit(0.S(32.W))
  val rsqrt = RegInit(0.S(32.W))

  // Accept control signals
  io.ctrl_reduce_i.ready := state === idle
  when(io.ctrl_reduce_i.fire) {
    iter_reg := io.ctrl_reduce_i.bits.iter
    norm_dim_reg := io.ctrl_reduce_i.bits.norm_dim
    state := accumulate
    batch_cnt := 0.U
    cur_batch := 0.U
  }

  // Default output
  io.ld_reduce_i.ready := false.B
  io.reduce_norm_o.valid := false.B
  io.reduce_norm_o.bits.mean := 0.S
  io.reduce_norm_o.bits.rsqrt := 0.S
  io.reduce_norm_o.bits.batch_idx := 0.U

  // Epsilon value (1e-5 in Q16.16 format)
  val epsilon = (0.00001 * (1 << 16)).toInt.S(32.W)

  // State machine logic
  switch(state) {
    is(idle) {
      sum_acc := 0.S
      sum_sq_acc := 0.S
      vec_cnt := 0.U
    }

    is(accumulate) {
      io.ld_reduce_i.ready := true.B

      when(io.ld_reduce_i.fire) {
        // Accumulate sum and sum of squares
        val vec_data = io.ld_reduce_i.bits.data
        var vec_sum = 0.S(48.W)
        var vec_sum_sq = 0.S(64.W)

        // Tree reduction for sum
        for (i <- 0 until b.veclane) {
          vec_sum = vec_sum + vec_data(i)
          vec_sum_sq = vec_sum_sq + (vec_data(i).asSInt * vec_data(i).asSInt)
        }

        sum_acc := sum_acc + vec_sum
        sum_sq_acc := sum_sq_acc + vec_sum_sq
        vec_cnt := vec_cnt + 1.U

        // Check if this is the last vector of the batch
        when(io.ld_reduce_i.bits.is_last) {
          state := compute
          cur_batch := io.ld_reduce_i.bits.batch_idx
        }
      }
    }

    is(compute) {
      // Compute mean: μ = sum / (norm_dim * 16)
      val total_elements = (norm_dim_reg << 4).asSInt  // norm_dim * 16
      val mean_val = (sum_acc << 16) / total_elements  // Q16.16 format
      mean := mean_val.asSInt

      // Compute E[x²]
      val mean_sq = (sum_sq_acc << 16) / total_elements

      // Compute variance: σ² = E[x²] - μ²
      val variance = mean_sq - ((mean_val * mean_val) >> 16)

      // Compute rsqrt: 1/√(σ² + ε)
      // Simplified implementation using lookup table approximation
      val var_with_eps = variance + epsilon
      val rsqrt_val = rsqrtApprox(var_with_eps)
      rsqrt := rsqrt_val

      // Send to normalize unit
      io.reduce_norm_o.valid := true.B
      io.reduce_norm_o.bits.mean := mean
      io.reduce_norm_o.bits.rsqrt := rsqrt
      io.reduce_norm_o.bits.batch_idx := cur_batch

      when(io.reduce_norm_o.ready) {
        // Reset for next batch
        sum_acc := 0.S
        sum_sq_acc := 0.S
        vec_cnt := 0.U
        batch_cnt := batch_cnt + 1.U

        when(batch_cnt === iter_reg - 1.U) {
          state := done
        }.otherwise {
          state := accumulate
        }
      }
    }

    is(done) {
      // Stay in done state
    }
  }

  // RSQRT approximation function (simplified LUT-based)
  def rsqrtApprox(x: SInt): SInt = {
    // For simplicity, use a piecewise linear approximation
    // In real implementation, use proper LUT or Newton-Raphson
    // This is a placeholder that returns 1.0 in Q16.16 format
    val one_q16 = (1 << 16).S(32.W)

    // Very simplified approximation: rsqrt ≈ 1/√x
    // For x in Q16.16 format, approximate sqrt(x) and invert
    // This should be replaced with proper LUT or iterative algorithm
    Mux(x > 0.S, one_q16, one_q16)
  }
}
