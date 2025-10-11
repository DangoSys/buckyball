package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramReadReq, SramReadResp}

class SoftmaxLoadUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_ld_i = Flipped(Decoupled(new SMCtrlLdReq))

    // SRAM read interface
    val sramReadReq  = Vec(b.sp_banks, Decoupled(new SramReadReq(b.spad_bank_entries)))
    val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(b.spad_w))))

    // ACC read interface
    val accReadReq   = Vec(b.acc_banks, Decoupled(new SramReadReq(b.acc_bank_entries)))
    val accReadResp  = Vec(b.acc_banks, Flipped(Decoupled(new SramReadResp(b.acc_w))))

    // Output to FindMax unit
    val ld_findmax_o = Decoupled(new SMLdFindMaxReq)
  })

  // State machine
  val idle :: loading :: Nil = Enum(2)
  val state = RegInit(idle)

  // Control registers
  val op1_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val op1_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter          = RegInit(0.U(10.W))
  val is_acc        = RegInit(false.B)
  val dim_len       = RegInit(0.U(10.W))
  val batch         = RegInit(0.U(10.W))

  // Address generation
  val vec_cnt    = RegInit(0.U(12.W))
  val batch_cnt  = RegInit(0.U(10.W))

  // Accept control request
  io.ctrl_ld_i.ready := state === idle
  when(io.ctrl_ld_i.fire) {
    op1_bank      := io.ctrl_ld_i.bits.op1_bank
    op1_bank_addr := io.ctrl_ld_i.bits.op1_bank_addr
    iter          := io.ctrl_ld_i.bits.iter
    is_acc        := io.ctrl_ld_i.bits.is_acc
    dim_len       := io.ctrl_ld_i.bits.dim_len
    batch         := io.ctrl_ld_i.bits.batch
    vec_cnt       := 0.U
    batch_cnt     := 0.U
    state         := loading
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
  io.ld_findmax_o.valid := false.B
  io.ld_findmax_o.bits.data := VecInit(Seq.fill(b.veclane)(0.S(32.W)))
  io.ld_findmax_o.bits.vec_idx := 0.U
  io.ld_findmax_o.bits.batch_idx := 0.U

  // Read state machine
  when(state === loading) {
    val addr = op1_bank_addr + vec_cnt

    when(!is_acc) {
      // SRAM read
      val bank_id = op1_bank
      when(bank_id < b.sp_banks.U) {
        io.sramReadReq(bank_id).valid := true.B
        io.sramReadReq(bank_id).bits.addr := addr
        io.sramReadReq(bank_id).bits.fromDMA := false.B

        // Accept response
        io.sramReadResp(bank_id).ready := io.ld_findmax_o.ready

        when(io.sramReadResp(bank_id).fire) {
          // Convert INT8 to INT32 with sign extension
          val sram_vec = io.sramReadResp(bank_id).bits.data.asTypeOf(Vec(b.veclane, SInt(8.W)))
          io.ld_findmax_o.valid := true.B
          io.ld_findmax_o.bits.data := VecInit(sram_vec.map(_.asSInt))
          io.ld_findmax_o.bits.vec_idx := vec_cnt
          io.ld_findmax_o.bits.batch_idx := batch_cnt

          when(io.ld_findmax_o.fire) {
            vec_cnt := vec_cnt + 1.U

            // Check if finished current batch
            when(vec_cnt + 1.U >= iter) {
              batch_cnt := batch_cnt + 1.U

              // Check if all batches done
              when(batch_cnt + 1.U >= batch) {
                state := idle
                vec_cnt := 0.U
                batch_cnt := 0.U
              }.otherwise {
                vec_cnt := 0.U
              }
            }
          }
        }
      }
    }.otherwise {
      // ACC read
      val bank_id = op1_bank
      when(bank_id < b.acc_banks.U) {
        io.accReadReq(bank_id).valid := true.B
        io.accReadReq(bank_id).bits.addr := addr
        io.accReadReq(bank_id).bits.fromDMA := false.B

        // Accept response
        io.accReadResp(bank_id).ready := io.ld_findmax_o.ready

        when(io.accReadResp(bank_id).fire) {
          // Already INT32
          val acc_vec = io.accReadResp(bank_id).bits.data.asTypeOf(Vec(b.veclane, SInt(32.W)))
          io.ld_findmax_o.valid := true.B
          io.ld_findmax_o.bits.data := acc_vec
          io.ld_findmax_o.bits.vec_idx := vec_cnt
          io.ld_findmax_o.bits.batch_idx := batch_cnt

          when(io.ld_findmax_o.fire) {
            vec_cnt := vec_cnt + 1.U

            // Check if finished current batch
            when(vec_cnt + 1.U >= iter) {
              batch_cnt := batch_cnt + 1.U

              // Check if all batches done
              when(batch_cnt + 1.U >= batch) {
                state := idle
                vec_cnt := 0.U
                batch_cnt := 0.U
              }.otherwise {
                vec_cnt := 0.U
              }
            }
          }
        }
      }
    }
  }
}
