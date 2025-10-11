package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class SoftmaxNormalizeUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_norm_i = Flipped(Decoupled(new SMCtrlNormReq))
    val expsum_norm_i = Flipped(Decoupled(new SMExpSumNormReq))
    val expsum_data_i = Flipped(Decoupled(new SMLdFindMaxReq))
    val norm_st_o = Decoupled(new SMNormStReq)
  })

  // State machine
  val idle :: wait_sum :: normalizing :: Nil = Enum(3)
  val state = RegInit(idle)

  // Control registers
  val iter     = RegInit(0.U(10.W))
  val dim_len  = RegInit(0.U(10.W))
  val batch    = RegInit(0.U(10.W))
  val log_mode = RegInit(false.B)

  // Sum value from ExpSum unit
  val sum_exp  = RegInit(0.U(32.W))
  val batch_idx = RegInit(0.U(10.W))

  // Tracking
  val vec_cnt = RegInit(0.U(12.W))

  // Accept control request
  io.ctrl_norm_i.ready := state === idle
  when(io.ctrl_norm_i.fire) {
    iter     := io.ctrl_norm_i.bits.iter
    dim_len  := io.ctrl_norm_i.bits.dim_len
    batch    := io.ctrl_norm_i.bits.batch
    log_mode := io.ctrl_norm_i.bits.log_mode
    vec_cnt  := 0.U
    state    := wait_sum
  }

  // Wait for sum value
  io.expsum_norm_i.ready := state === wait_sum
  when(io.expsum_norm_i.fire) {
    sum_exp   := io.expsum_norm_i.bits.sum_exp
    batch_idx := io.expsum_norm_i.bits.batch_idx
    state     := normalizing
  }

  // Normalize
  io.expsum_data_i.ready := state === normalizing

  val norm_vec = Wire(Vec(b.veclane, SInt(32.W)))
  // Initialize default values
  for (i <- 0 until b.veclane) {
    norm_vec(i) := 0.S
  }

  when(io.expsum_data_i.fire) {
    // Normalize: exp(x) / sum_exp
    for (i <- 0 until b.veclane) {
      val exp_val = io.expsum_data_i.bits.data(i).asUInt

      when(!log_mode) {
        // Standard Softmax: exp(x) / sum
        // Use fixed-point division
        // Result = (exp_val << 16) / sum_exp
        val scaled = (exp_val << 16)
        val quotient = Mux(sum_exp =/= 0.U, scaled / sum_exp, 0.U)
        norm_vec(i) := quotient.asSInt
      }.otherwise {
        // LogSoftmax: log(exp(x) / sum) = x - log(sum)
        // Approximation: log(x) ≈ x (simplified)
        // TODO: Implement proper log approximation
        val log_sum = sum_exp >> 8  // Rough approximation
        norm_vec(i) := io.expsum_data_i.bits.data(i) - log_sum.asSInt
      }
    }

    vec_cnt := vec_cnt + 1.U
  }

  // Output to store unit
  io.norm_st_o.valid := io.expsum_data_i.fire
  io.norm_st_o.bits.data := norm_vec
  io.norm_st_o.bits.vec_idx := vec_cnt
  io.norm_st_o.bits.batch_idx := batch_idx
  io.norm_st_o.bits.is_last := (vec_cnt + 1.U >= iter)

  when(io.norm_st_o.fire && io.norm_st_o.bits.is_last) {
    batch_idx := batch_idx + 1.U

    // Check if all batches done
    when(batch_idx + 1.U >= batch) {
      state := idle
      batch_idx := 0.U
    }.otherwise {
      // Reset for next batch
      vec_cnt := 0.U
      state := wait_sum
    }
  }
}
