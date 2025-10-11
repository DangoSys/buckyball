package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramWriteIO, SramWriteReq}

class SoftmaxStoreUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new SMCtrlStReq))
    val norm_st_i = Flipped(Decoupled(new SMNormStReq))

    // SRAM write interface
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

    // ACC write interface
    val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    // Completion signal
    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  // State machine
  val idle :: storing :: Nil = Enum(2)
  val state = RegInit(idle)

  // Control registers
  val wr_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter         = RegInit(0.U(10.W))
  val is_acc       = RegInit(false.B)
  val dim_len      = RegInit(0.U(10.W))

  // Tracking
  val vec_cnt = RegInit(0.U(12.W))

  // Accept control request
  io.ctrl_st_i.ready := state === idle
  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := io.ctrl_st_i.bits.iter
    is_acc       := io.ctrl_st_i.bits.is_acc
    dim_len      := io.ctrl_st_i.bits.dim_len
    vec_cnt      := 0.U
    state        := storing
  }

  // Default outputs - initialize all write interfaces
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

  // Accept data from normalize unit
  io.norm_st_i.ready := state === storing

  when(io.norm_st_i.fire) {
    val addr = wr_bank_addr + vec_cnt

    when(!is_acc) {
      // Write to SRAM (INT8)
      val sram_data = Wire(UInt(b.spad_w.W))
      val data_bytes = Wire(Vec(b.veclane, UInt(8.W)))
      for (i <- 0 until b.veclane) {
        // Convert INT32 to INT8 (with saturation)
        val val32 = io.norm_st_i.bits.data(i)
        val clamped = Mux(val32 > 127.S, 127.S,
                      Mux(val32 < (-128).S, (-128).S, val32))
        data_bytes(i) := clamped.asUInt(7, 0)
      }
      sram_data := data_bytes.asUInt

      val bank_id = wr_bank
      when(bank_id < b.sp_banks.U) {
        io.sramWrite(bank_id).req.valid := true.B
        io.sramWrite(bank_id).req.bits.addr := addr
        io.sramWrite(bank_id).req.bits.data := sram_data
        io.sramWrite(bank_id).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(true.B))
      }
    }.otherwise {
      // Write to ACC (INT32)
      val acc_data = Wire(UInt(b.acc_w.W))
      val data_words = Wire(Vec(b.veclane, UInt(32.W)))
      for (i <- 0 until b.veclane) {
        data_words(i) := io.norm_st_i.bits.data(i).asUInt
      }
      acc_data := data_words.asUInt

      val bank_id = wr_bank
      when(bank_id < b.acc_banks.U) {
        io.accWrite(bank_id).req.valid := true.B
        io.accWrite(bank_id).req.bits.addr := addr
        io.accWrite(bank_id).req.bits.data := acc_data
        io.accWrite(bank_id).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }
    }

    vec_cnt := vec_cnt + 1.U

    // Check if all writes done
    when(vec_cnt + 1.U >= iter) {
      state := idle
      vec_cnt := 0.U
    }
  }

  // Send completion signal when done
  io.cmdResp_o.valid := state === idle && vec_cnt === 0.U && RegNext(state === storing)
  io.cmdResp_o.bits.commit := true.B
}
