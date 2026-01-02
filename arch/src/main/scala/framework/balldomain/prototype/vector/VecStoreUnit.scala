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

  val wr_bank             = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank_addr        = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val iter                = RegInit(0.U(10.W))
  val iter_counter        = RegInit(0.U(10.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := (io.ctrl_st_i.bits.iter + 15.U(10.W)) & (~15.U(10.W))
    iter_counter := 0.U
    state        := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit and perform write-back
// -----------------------------------------------------------------------------
  io.ex_st_i.ready := state === busy
  io.bankWrite.foreach { acc =>
    acc.req.valid      := false.B
    acc.req.bits.addr  := 0.U
    acc.req.bits.data  := Cat(Seq.fill(accWidth / 8)(0.U(8.W)))
    acc.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    acc.req.bits.wmode := false.B // Default: direct write mode
    acc.resp.ready     := false.B // Default: not ready for response
  }
  val waddr = wr_bank_addr + iter_counter(log2Ceil(InputNum) - 1, 0)
  when(io.ex_st_i.fire) {
    // Use all outBW channels to share the data write
    // Each channel gets a portion of the data
    for (i <- 0 until outBW) {
      val elementsPerChannel = InputNum / outBW
      val startIdx           = i * elementsPerChannel
      val endIdx             = startIdx + elementsPerChannel - 1

      io.bankWrite(i).req.valid     := true.B
      io.bankWrite(i).req.bits.addr := wr_bank_addr + iter_counter(log2Ceil(InputNum) - 1, 0)

      // Pack corresponding elements into a UInt
      val channelData = Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
      io.bankWrite(i).req.bits.data  := channelData
      io.bankWrite(i).req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
      io.bankWrite(i).req.bits.wmode := true.B // Accumulate mode
    }
    iter_counter := iter_counter + 1.U
  }

  // Output wr_bank for bank_id setting
  io.wr_bank_o := wr_bank

// -----------------------------------------------------------------------------
// Reset iter counter, commit cmdResp, return to idle state
// -----------------------------------------------------------------------------
  when(state === busy && iter_counter >= iter) {
    state                    := idle
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.commit := false.B
  }

}
