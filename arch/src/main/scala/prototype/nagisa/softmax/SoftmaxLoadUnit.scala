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
  val total_vecs = iter

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
    io.sramReadResp(i).ready := true.B
  }
  for (i <- 0 until b.acc_banks) {
    io.accReadReq(i).valid := false.B
    io.accReadReq(i).bits.addr := 0.U
    io.accReadReq(i).bits.fromDMA := false.B
    io.accReadResp(i).ready := true.B
  }

  // Read state machine
  val read_pending = RegInit(false.B)
  val resp_pending = RegInit(false.B)

  when(state === loading) {
    when(!read_pending && vec_cnt < total_vecs) {
      // Issue read request
      val addr = op1_bank_addr + vec_cnt
      when(!is_acc) {
        // SRAM read
        io.sramReadReq(op1_bank).valid := true.B
        io.sramReadReq(op1_bank).bits.addr := addr
        io.sramReadReq(op1_bank).bits.fromDMA := false.B
        when(io.sramReadReq(op1_bank).fire) {
          read_pending := true.B
          resp_pending := true.B
        }
      }.otherwise {
        // ACC read
        io.accReadReq(op1_bank).valid := true.B
        io.accReadReq(op1_bank).bits.addr := addr
        io.accReadReq(op1_bank).bits.fromDMA := false.B
        when(io.accReadReq(op1_bank).fire) {
          read_pending := true.B
          resp_pending := true.B
        }
      }
    }
  }

  // Response handling
  val resp_data = Wire(UInt(512.W))
  val resp_valid = Wire(Bool())

  when(!is_acc) {
    resp_valid := io.sramReadResp(op1_bank).valid
    resp_data := io.sramReadResp(op1_bank).bits.data
  }.otherwise {
    resp_valid := io.accReadResp(op1_bank).valid
    resp_data := io.accReadResp(op1_bank).bits.data
  }

  // Convert data to vec of INT32
  val data_vec = Wire(Vec(b.veclane, SInt(32.W)))
  when(!is_acc) {
    // INT8 to INT32 conversion with sign extension
    val sram_vec = resp_data.asTypeOf(Vec(b.veclane, SInt(8.W)))
    for (i <- 0 until b.veclane) {
      data_vec(i) := sram_vec(i)
    }
  }.otherwise {
    // Already INT32
    val acc_vec = resp_data.asTypeOf(Vec(b.veclane, SInt(32.W)))
    data_vec := acc_vec
  }

  io.ld_findmax_o.valid := false.B
  io.ld_findmax_o.bits.data := data_vec
  io.ld_findmax_o.bits.vec_idx := vec_cnt
  io.ld_findmax_o.bits.batch_idx := batch_cnt

  when(resp_pending && resp_valid) {
    io.ld_findmax_o.valid := true.B
    when(io.ld_findmax_o.fire) {
      resp_pending := false.B
      read_pending := false.B
      vec_cnt := vec_cnt + 1.U

      // Check if finished current batch
      when(vec_cnt + 1.U >= total_vecs) {
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
