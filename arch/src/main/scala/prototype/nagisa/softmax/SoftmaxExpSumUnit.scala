package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class SoftmaxExpSumUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_expsum_i = Flipped(Decoupled(new SMCtrlExpSumReq))
    val findmax_expsum_i = Flipped(Decoupled(new SMFindMaxExpSumReq))
    val findmax_data_i = Flipped(Decoupled(new SMLdFindMaxReq))  // Data from findmax
    val expsum_norm_o = Decoupled(new SMExpSumNormReq)
    val expsum_data_o = Decoupled(new SMLdFindMaxReq)  // Pass exp data to normalize
  })

  // State machine
  val idle :: wait_max :: computing :: complete :: Nil = Enum(4)
  val state = RegInit(idle)

  // Control registers
  val iter     = RegInit(0.U(10.W))
  val dim_len  = RegInit(0.U(10.W))
  val batch    = RegInit(0.U(10.W))

  // Max value from FindMax unit
  val max_val  = RegInit(0.S(32.W))
  val batch_idx = RegInit(0.U(10.W))

  // Sum of exp values
  val sum_exp = RegInit(0.U(32.W))

  // Tracking
  val vec_cnt = RegInit(0.U(12.W))

  // Exp result buffer
  val exp_buffer = Mem(1024, Vec(b.veclane, SInt(32.W)))
  val exp_buffer_wr_ptr = RegInit(0.U(10.W))

  // Accept control request
  io.ctrl_expsum_i.ready := state === idle
  when(io.ctrl_expsum_i.fire) {
    iter     := io.ctrl_expsum_i.bits.iter
    dim_len  := io.ctrl_expsum_i.bits.dim_len
    batch    := io.ctrl_expsum_i.bits.batch
    vec_cnt  := 0.U
    sum_exp  := 0.U
    exp_buffer_wr_ptr := 0.U
    state    := wait_max
  }

  // Wait for max value
  io.findmax_expsum_i.ready := state === wait_max
  when(io.findmax_expsum_i.fire) {
    max_val   := io.findmax_expsum_i.bits.max_val
    batch_idx := io.findmax_expsum_i.bits.batch_idx
    state     := computing
  }

  // Compute exp and accumulate
  io.findmax_data_i.ready := state === computing

  when(io.findmax_data_i.fire) {
    // Compute exp(x - max) for each element
    val exp_vec = Wire(Vec(b.veclane, SInt(32.W)))
    for (i <- 0 until b.veclane) {
      val shifted = io.findmax_data_i.bits.data(i) - max_val
      // Simple approximation: exp(x) ≈ 1 + x (for small x)
      // TODO: Use proper exp approximation (LUT or polynomial)
      val exp_val = Mux(shifted < -16.S, 0.S,
                    Mux(shifted > 16.S, 65536.S,  // Scale factor
                    (shifted << 12).asSInt))  // Q20.12 fixed point
      exp_vec(i) := exp_val

      // Accumulate sum
      sum_exp := sum_exp + exp_val.asUInt
    }

    // Store exp results in buffer
    exp_buffer.write(exp_buffer_wr_ptr, exp_vec)
    exp_buffer_wr_ptr := exp_buffer_wr_ptr + 1.U

    vec_cnt := vec_cnt + 1.U

    // Check if finished
    when(vec_cnt + 1.U >= iter) {
      state := complete
    }
  }

  // Output sum of exp
  io.expsum_norm_o.valid := state === complete
  io.expsum_norm_o.bits.sum_exp := sum_exp
  io.expsum_norm_o.bits.batch_idx := batch_idx

  when(io.expsum_norm_o.fire) {
    batch_idx := batch_idx + 1.U

    // Check if all batches done
    when(batch_idx + 1.U >= batch) {
      state := idle
      batch_idx := 0.U
    }.otherwise {
      // Reset for next batch
      vec_cnt := 0.U
      sum_exp := 0.U
      exp_buffer_wr_ptr := 0.U
      state := wait_max
    }
  }

  // Pass exp data to normalize unit
  val exp_read_ptr = RegInit(0.U(10.W))
  val exp_read_batch = RegInit(0.U(10.W))
  val exp_reading = RegInit(false.B)

  when(state === complete && !exp_reading) {
    exp_reading := true.B
    exp_read_ptr := 0.U
  }

  io.expsum_data_o.valid := exp_reading && exp_read_ptr < iter
  io.expsum_data_o.bits.data := exp_buffer.read(exp_read_ptr)
  io.expsum_data_o.bits.vec_idx := exp_read_ptr
  io.expsum_data_o.bits.batch_idx := exp_read_batch

  when(io.expsum_data_o.fire) {
    exp_read_ptr := exp_read_ptr + 1.U

    when(exp_read_ptr + 1.U >= iter) {
      exp_read_batch := exp_read_batch + 1.U

      when(exp_read_batch + 1.U >= batch) {
        exp_reading := false.B
        exp_read_ptr := 0.U
        exp_read_batch := 0.U
      }.otherwise {
        exp_read_ptr := 0.U
      }
    }
  }
}
