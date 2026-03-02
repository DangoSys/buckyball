package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.SramWriteIO
import framework.top.GlobalConfig
import framework.balldomain.prototype.vector.configs.VectorBallParam

class ctrl_st_req(b: GlobalConfig) extends Bundle {
  val wr_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val iter         = UInt(10.W)
}

class ex_st_req(b: GlobalConfig) extends Bundle {
  val config   = VectorBallParam()
  val InputNum = config.lane
  val accWidth = config.outputWidth
  val rst      = Vec(InputNum, UInt(accWidth.W))
  val iter     = UInt(10.W)
}

@instantiable
class VecStoreUnit(val b: GlobalConfig) extends Module {
  val config   = VectorBallParam()
  val InputNum = config.lane
  val accWidth = config.outputWidth

  // Get bandwidth from config (use first VecBall mapping)
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
    .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new ctrl_st_req(b)))
    val ex_st_i   = Flipped(Decoupled(new ex_st_req(b)))
    val bankWrite = Vec(outBW, Flipped(new SramWriteIO(b)))
    val wr_bank_o = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  val wr_bank                                   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank_addr                              = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val iter                                      = RegInit(0.U(10.W))
  val iter_counter                              = RegInit(0.U(10.W))
  val idle :: busy :: write :: wait_last :: Nil = Enum(4)
  val state                                     = RegInit(idle)

  // Register to hold data from EX unit
  val data_valid = RegInit(false.B)
  val data_addr  = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val data_vec   = Reg(Vec(InputNum, UInt(accWidth.W)))

  // Track which channels have fired
  val channel_fired = RegInit(VecInit(Seq.fill(outBW)(false.B)))

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := (io.ctrl_st_i.bits.iter + 15.U(10.W)) & (~15.U(10.W))
    iter_counter := 0.U
    data_valid   := false.B
    state        := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit
// -----------------------------------------------------------------------------
  io.ex_st_i.ready := (state === busy || state === write) && !data_valid

  when(io.ex_st_i.fire) {
    // Latch data
    data_valid := true.B
    data_addr  := wr_bank_addr + iter_counter
    data_vec   := io.ex_st_i.bits.rst
    state      := write
    // Reset channel_fired when receiving new data
    for (i <- 0 until outBW) {
      channel_fired(i) := false.B
    }
  }

// -----------------------------------------------------------------------------
// Write data to banks
// -----------------------------------------------------------------------------
  // Default values for bankWrite
  io.bankWrite.foreach { acc =>
    acc.req.valid      := false.B
    acc.req.bits.addr  := 0.U
    acc.req.bits.data  := 0.U
    acc.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    acc.req.bits.wmode := false.B
    acc.resp.ready     := true.B
  }

  when(state === write && data_valid) {
    val all_fired = channel_fired.reduce(_ && _)

    for (i <- 0 until outBW) {
      val elementsPerChannel = InputNum / outBW
      val startIdx           = i * elementsPerChannel
      val endIdx             = startIdx + elementsPerChannel - 1

      // Only send request if this channel hasn't fired yet
      when(!channel_fired(i)) {
        io.bankWrite(i).req.valid      := true.B
        io.bankWrite(i).req.bits.addr  := data_addr
        io.bankWrite(i).req.bits.data  := Cat(data_vec.slice(startIdx, endIdx + 1).reverse)
        io.bankWrite(i).req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
        io.bankWrite(i).req.bits.wmode := true.B // Accumulator mode

        // Mark as fired when handshake completes
        when(io.bankWrite(i).req.ready) {
          channel_fired(i) := true.B
        }
      }
    }

    when(all_fired) {
      data_valid   := false.B
      iter_counter := iter_counter + 1.U

      // Reset channel_fired for next iter
      for (i <- 0 until outBW) {
        channel_fired(i) := false.B
      }

      // Check if this is the last iter
      when(iter_counter + 1.U >= iter) {
        state := wait_last
      }.otherwise {
        state := busy
      }
    }
  }

  // Output wr_bank for bank_id setting
  io.wr_bank_o := wr_bank

// -----------------------------------------------------------------------------
// Wait one cycle for last write to complete, then return to idle
// -----------------------------------------------------------------------------
  when(state === wait_last) {
    state                    := idle
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.commit := false.B
  }

}
