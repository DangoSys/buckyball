package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class LayerNormNormalizeUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_norm_i = Flipped(Decoupled(new LNCtrlNormReq))
    val ld_reduce_i = Flipped(Decoupled(new LNLdReduceReq))  // Raw data from load
    val reduce_norm_i = Flipped(Decoupled(new LNReduceNormReq))  // Mean and rsqrt
    val ld_norm_param_i = Flipped(Valid(new LNLdNormParam))  // Gamma and beta
    val norm_st_o = Decoupled(new LNNormStReq)
  })

  // State machine
  val idle :: wait_stats :: normalize :: done :: Nil = Enum(4)
  val state = RegInit(idle)

  // Control registers
  val iter_reg = RegInit(0.U(10.W))
  val norm_dim_reg = RegInit(0.U(12.W))
  val use_affine_reg = RegInit(false.B)

  // Statistics from reduce unit
  val mean_reg = RegInit(0.S(32.W))
  val rsqrt_reg = RegInit(0.S(32.W))
  val batch_cnt = RegInit(0.U(10.W))
  val cur_batch = RegInit(0.U(10.W))

  // Data buffer (stores raw data for current batch)
  val data_buffer = Reg(Vec(128, Vec(b.veclane, SInt(32.W))))  // Buffer for max 128 vectors
  val vec_cnt = RegInit(0.U(12.W))
  val buffered_vecs = RegInit(0.U(12.W))

  // Gamma and beta buffers
  val gamma_buffer = Reg(Vec(128, Vec(b.veclane, SInt(32.W))))
  val beta_buffer = Reg(Vec(128, Vec(b.veclane, SInt(32.W))))

  // Accept control signals
  io.ctrl_norm_i.ready := state === idle
  when(io.ctrl_norm_i.fire) {
    iter_reg := io.ctrl_norm_i.bits.iter
    norm_dim_reg := io.ctrl_norm_i.bits.norm_dim
    use_affine_reg := io.ctrl_norm_i.bits.use_affine
    state := wait_stats
    batch_cnt := 0.U
    vec_cnt := 0.U
    buffered_vecs := 0.U
  }

  // Default outputs
  io.ld_reduce_i.ready := false.B
  io.reduce_norm_i.ready := false.B
  io.norm_st_o.valid := false.B
  io.norm_st_o.bits.data := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.norm_st_o.bits.batch_idx := 0.U
  io.norm_st_o.bits.vec_idx := 0.U
  io.norm_st_o.bits.is_last := false.B

  // State machine logic
  switch(state) {
    is(idle) {
      buffered_vecs := 0.U
      vec_cnt := 0.U
    }

    is(wait_stats) {
      // Buffer incoming data
      io.ld_reduce_i.ready := true.B
      when(io.ld_reduce_i.fire) {
        val vec_idx = io.ld_reduce_i.bits.vec_idx
        data_buffer(vec_idx) := io.ld_reduce_i.bits.data
        when(io.ld_reduce_i.bits.is_last) {
          buffered_vecs := vec_idx + 1.U
          cur_batch := io.ld_reduce_i.bits.batch_idx
        }
      }

      // Wait for statistics
      io.reduce_norm_i.ready := true.B
      when(io.reduce_norm_i.fire) {
        mean_reg := io.reduce_norm_i.bits.mean
        rsqrt_reg := io.reduce_norm_i.bits.rsqrt
        state := normalize
        vec_cnt := 0.U
      }
    }

    is(normalize) {
      // Normalize each vector
      when(vec_cnt < buffered_vecs) {
        val vec_data = data_buffer(vec_cnt)
        val normalized = Wire(Vec(b.veclane, SInt(32.W)))

        // Normalize: x̂ = (x - μ) * rsqrt
        for (i <- 0 until b.veclane) {
          val centered = vec_data(i) - mean_reg
          // Fixed-point multiplication: Q16.16 * Q16.16 = Q32.32, shift right 16 to get Q16.16
          val norm_val = ((centered.asSInt * rsqrt_reg.asSInt) >> 16).asSInt

          // Apply affine transformation if enabled
          val output_val = Mux(use_affine_reg,
            {
              // y = γ * x̂ + β
              val gamma_val = gamma_buffer(vec_cnt)(i)
              val beta_val = beta_buffer(vec_cnt)(i)
              val scaled = ((norm_val.asSInt * gamma_val.asSInt) >> 16).asSInt
              (scaled + beta_val).asSInt
            },
            norm_val
          )

          normalized(i) := output_val
        }

        // Send to store unit
        io.norm_st_o.valid := true.B
        io.norm_st_o.bits.data := normalized
        io.norm_st_o.bits.batch_idx := cur_batch
        io.norm_st_o.bits.vec_idx := vec_cnt
        io.norm_st_o.bits.is_last := (vec_cnt === buffered_vecs - 1.U)

        when(io.norm_st_o.ready) {
          when(vec_cnt === buffered_vecs - 1.U) {
            // Finished normalizing current batch
            batch_cnt := batch_cnt + 1.U
            when(batch_cnt === iter_reg - 1.U) {
              state := done
            }.otherwise {
              state := wait_stats
              vec_cnt := 0.U
              buffered_vecs := 0.U
            }
          }.otherwise {
            vec_cnt := vec_cnt + 1.U
          }
        }
      }
    }

    is(done) {
      // Stay in done state
    }
  }
}
