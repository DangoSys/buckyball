package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}

class LayerNormStoreUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new LNCtrlStReq))
    val norm_st_i = Flipped(Decoupled(new LNNormStReq))

    // Memory write interfaces
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  // State machine
  val idle :: store :: done :: Nil = Enum(3)
  val state = RegInit(idle)

  // Control registers
  val wr_bank = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter_reg = RegInit(0.U(10.W))
  val is_acc_reg = RegInit(false.B)
  val norm_dim_reg = RegInit(0.U(12.W))

  // Counters
  val batch_cnt = RegInit(0.U(10.W))
  val vec_cnt = RegInit(0.U(12.W))

  // Accept control signals
  io.ctrl_st_i.ready := state === idle
  when(io.ctrl_st_i.fire) {
    wr_bank := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter_reg := io.ctrl_st_i.bits.iter
    is_acc_reg := io.ctrl_st_i.bits.is_acc
    norm_dim_reg := io.ctrl_st_i.bits.norm_dim
    state := store
    batch_cnt := 0.U
    vec_cnt := 0.U
  }

  // Default outputs
  for (i <- 0 until b.sp_banks) {
    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(false.B))
  }
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits.addr := 0.U
    io.accWrite(i).req.bits.data := 0.U
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(false.B))
  }
  io.norm_st_i.ready := false.B
  io.cmdResp_o.valid := false.B
  io.cmdResp_o.bits.commit := false.B

  // State machine logic
  switch(state) {
    is(idle) {
      // Wait for control signal
    }

    is(store) {
      io.norm_st_i.ready := true.B

      when(io.norm_st_i.fire) {
        val data_vec = io.norm_st_i.bits.data
        val batch_idx = io.norm_st_i.bits.batch_idx
        val vec_idx = io.norm_st_i.bits.vec_idx
        val is_last = io.norm_st_i.bits.is_last

        // Calculate write address
        val addr = wr_bank_addr + batch_idx * norm_dim_reg + vec_idx

        // Convert INT32 data to target format
        when(is_acc_reg) {
          // ACC mode: write 512-bit data (16xINT32)
          val write_data = Wire(UInt(b.acc_w.W))
          val data_bits = Wire(Vec(b.veclane, UInt(32.W)))
          for (i <- 0 until b.veclane) {
            data_bits(i) := data_vec(i).asUInt
          }
          write_data := data_bits.asUInt

          val bank_id = wr_bank
          when(bank_id < b.acc_banks.U) {
            io.accWrite(bank_id).req.valid := true.B
            io.accWrite(bank_id).req.bits.addr := addr
            io.accWrite(bank_id).req.bits.data := write_data
            io.accWrite(bank_id).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
          }
        }.otherwise {
          // SRAM mode: write 128-bit data (16xINT8)
          // Clamp to INT8 range [-128, 127]
          val write_data = Wire(UInt(b.spad_w.W))
          val data_bits = Wire(Vec(b.veclane, UInt(8.W)))
          for (i <- 0 until b.veclane) {
            val clamped = Mux(data_vec(i) > 127.S, 127.S,
                          Mux(data_vec(i) < (-128).S, (-128).S, data_vec(i)))
            data_bits(i) := clamped.asUInt(7, 0)
          }
          write_data := data_bits.asUInt

          val bank_id = wr_bank
          when(bank_id < b.sp_banks.U) {
            io.sramWrite(bank_id).req.valid := true.B
            io.sramWrite(bank_id).req.bits.addr := addr
            io.sramWrite(bank_id).req.bits.data := write_data
            io.sramWrite(bank_id).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(true.B))
          }
        }

        // Update counters
        vec_cnt := vec_idx + 1.U

        when(is_last) {
          // Finished current batch
          batch_cnt := batch_idx + 1.U
          vec_cnt := 0.U

          when(batch_idx === iter_reg - 1.U) {
            // All batches done
            state := done
          }
        }
      }
    }

    is(done) {
      // Send completion response
      io.cmdResp_o.valid := true.B
      io.cmdResp_o.bits.commit := true.B
      state := idle
    }
  }
}
