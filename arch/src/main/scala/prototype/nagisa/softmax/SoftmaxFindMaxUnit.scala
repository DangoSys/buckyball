package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import chisel3.experimental.BundleLiterals._

class SoftmaxFindMaxUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_findmax_i = Flipped(Decoupled(new SMCtrlFindMaxReq))
    val ld_findmax_i = Flipped(Decoupled(new SMLdFindMaxReq))
    val findmax_expsum_o = Decoupled(new SMFindMaxExpSumReq)
    val findmax_norm_o = Decoupled(new SMLdFindMaxReq)  // Pass data to normalize
  })

  // State machine
  val idle :: finding :: complete :: Nil = Enum(3)
  val state = RegInit(idle)

  // Control registers
  val iter     = RegInit(0.U(10.W))
  val dim_len  = RegInit(0.U(10.W))
  val batch    = RegInit(0.U(10.W))

  // Tracking
  val vec_cnt   = RegInit(0.U(12.W))
  val batch_cnt = RegInit(0.U(10.W))

  // Max value register per batch
  val max_val = RegInit((-2147483648).S(32.W))  // INT32_MIN

  // Data buffer - store all vectors for later use
  val data_buffer = Mem(1024, Vec(b.veclane, SInt(32.W)))
  val data_buffer_wr_ptr = RegInit(0.U(10.W))

  // Accept control request
  io.ctrl_findmax_i.ready := state === idle
  when(io.ctrl_findmax_i.fire) {
    iter     := io.ctrl_findmax_i.bits.iter
    dim_len  := io.ctrl_findmax_i.bits.dim_len
    batch    := io.ctrl_findmax_i.bits.batch
    vec_cnt  := 0.U
    batch_cnt := 0.U
    max_val  := (-2147483648).S(32.W)
    data_buffer_wr_ptr := 0.U
    state    := finding
  }

  // Find max in input data
  io.ld_findmax_i.ready := state === finding

  when(io.ld_findmax_i.fire) {
    // Store data in buffer
    data_buffer.write(data_buffer_wr_ptr, io.ld_findmax_i.bits.data)
    data_buffer_wr_ptr := data_buffer_wr_ptr + 1.U

    // Find max across all elements in the vector
    val vec_max = io.ld_findmax_i.bits.data.reduce((a, b) => Mux(a > b, a, b))

    // Update global max
    when(vec_max > max_val) {
      max_val := vec_max
    }

    vec_cnt := vec_cnt + 1.U

    // Check if finished current batch
    when(vec_cnt + 1.U >= iter) {
      state := complete
    }
  }

  // Output max value
  io.findmax_expsum_o.valid := state === complete
  io.findmax_expsum_o.bits.max_val := max_val
  io.findmax_expsum_o.bits.batch_idx := batch_cnt

  when(io.findmax_expsum_o.fire) {
    batch_cnt := batch_cnt + 1.U

    // Check if all batches done
    when(batch_cnt + 1.U >= batch) {
      state := idle
      batch_cnt := 0.U
    }.otherwise {
      // Reset for next batch
      vec_cnt := 0.U
      max_val := (-2147483648).S(32.W)
      state := finding
    }
  }

  // Pass data to normalize unit (read from buffer)
  val data_read_ptr = RegInit(0.U(10.W))
  val data_read_batch = RegInit(0.U(10.W))
  val data_reading = RegInit(false.B)

  // Start reading when we have completed finding max
  when(state === complete && !data_reading) {
    data_reading := true.B
    data_read_ptr := 0.U
  }

  io.findmax_norm_o.valid := data_reading && data_read_ptr < iter
  io.findmax_norm_o.bits.data := data_buffer.read(data_read_ptr)
  io.findmax_norm_o.bits.vec_idx := data_read_ptr
  io.findmax_norm_o.bits.batch_idx := data_read_batch

  when(io.findmax_norm_o.fire) {
    data_read_ptr := data_read_ptr + 1.U

    when(data_read_ptr + 1.U >= iter) {
      data_read_batch := data_read_batch + 1.U

      when(data_read_batch + 1.U >= batch) {
        data_reading := false.B
        data_read_ptr := 0.U
        data_read_batch := 0.U
      }.otherwise {
        data_read_ptr := 0.U
      }
    }
  }
}
