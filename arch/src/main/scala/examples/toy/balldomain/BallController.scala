package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import prototype.matrix._
import prototype.vector._
import prototype.Im2col
import examples.toy.balldomain.rs.{BallIssueInterface, BallCommitInterface}
// import framework.builtin.frontend.rs.{ReservationStationIssue, ReservationStationComplete, BuckyBallCmd}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class BallController(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  
  val io = IO(new Bundle {
    val cmdReq  = Flipped(new BallIssueInterface)
    val cmdResp = Flipped(new BallCommitInterface)
    
    // 连接到Scratchpad 和 Accumulator 的SRAM读写接口
    val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  val BBFP_Control = Module(new BBFP_Control)
  val VecUnit = Module(new VecUnit)
  val Im2col = Module(new Im2col)

// -----------------------------------------------------------------------------
// Input Selector
// -----------------------------------------------------------------------------

  val sel      = WireInit(0.U(2.W))
  val sel_reg  = RegInit(0.U(2.W))

  val im2col :: vec_unit :: bbfp_unit :: Nil = Enum(3)
  val real_sel = WireInit(vec_unit)

  // ball1 (VecUnit) 和 ball2 (BBFP) 的选择逻辑
  val ball1_valid = io.cmdReq.ball1.valid
  val ball2_valid = io.cmdReq.ball2.valid
  val ball3_valid = io.cmdReq.ball3.valid

  sel := Mux(ball1_valid, vec_unit, Mux(ball2_valid, bbfp_unit, Mux(ball3_valid, im2col, 0.U)))

  when (ball1_valid || ball2_valid || ball3_valid) {
    sel_reg := sel
  }.otherwise {
    sel_reg := sel_reg
  }
  real_sel := Mux(ball1_valid || ball2_valid || ball3_valid, sel, sel_reg)

  VecUnit.io.cmdReq.valid := io.cmdReq.ball1.valid
  VecUnit.io.cmdReq.bits := io.cmdReq.ball1.bits
  io.cmdReq.ball1.ready := VecUnit.io.cmdReq.ready
  
  BBFP_Control.io.cmdReq.valid := io.cmdReq.ball2.valid  
  BBFP_Control.io.cmdReq.bits := io.cmdReq.ball2.bits
  io.cmdReq.ball2.ready := BBFP_Control.io.cmdReq.ready

  Im2col.io.cmdReq.valid := io.cmdReq.ball3.valid
  Im2col.io.cmdReq.bits := io.cmdReq.ball3.bits
  io.cmdReq.ball3.ready := Im2col.io.cmdReq.ready

// -----------------------------------------------------------------------------
// 默认赋值, sb chisel 识别不出来两种情况全覆盖了，故手动赋值
// -----------------------------------------------------------------------------
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := false.B
    io.sramWrite(i).req.valid       := false.B
    io.sramWrite(i).req.bits.addr   := 0.U
    io.sramWrite(i).req.bits.data   := 0.U
    io.sramWrite(i).req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(false.B))
  }
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).req.valid         := false.B
    io.accRead(i).req.bits.addr     := 0.U
    io.accRead(i).req.bits.fromDMA  := false.B
    io.accRead(i).resp.ready        := false.B
    io.accWrite(i).req.valid        := false.B
    io.accWrite(i).req.bits.addr    := 0.U
    io.accWrite(i).req.bits.data    := 0.U
    io.accWrite(i).req.bits.mask    := VecInit(Seq.fill(b.acc_mask_len)(false.B))
  }
  BBFP_Control.io.sramRead.foreach(_.req.ready := false.B)
  BBFP_Control.io.sramRead.foreach(_.resp.valid := false.B)
  BBFP_Control.io.sramRead.foreach(_.resp.bits.data := 0.U)
  BBFP_Control.io.sramRead.foreach(_.resp.bits.fromDMA := false.B)
  BBFP_Control.io.sramWrite.foreach(_.req.ready := false.B)
  BBFP_Control.io.accRead.foreach(_.req.ready := false.B)
  BBFP_Control.io.accRead.foreach(_.resp.valid := false.B)
  BBFP_Control.io.accRead.foreach(_.resp.bits.data := 0.U)
  BBFP_Control.io.accRead.foreach(_.resp.bits.fromDMA := false.B)
  BBFP_Control.io.accWrite.foreach(_.req.ready := false.B)

  VecUnit.io.sramRead.foreach(_.req.ready := false.B)
  VecUnit.io.sramRead.foreach(_.resp.valid := false.B)
  VecUnit.io.sramRead.foreach(_.resp.bits.data := 0.U)
  VecUnit.io.sramRead.foreach(_.resp.bits.fromDMA := false.B)
  VecUnit.io.sramWrite.foreach(_.req.ready := false.B)
  VecUnit.io.accRead.foreach(_.req.ready := false.B)
  VecUnit.io.accRead.foreach(_.resp.valid := false.B)
  VecUnit.io.accRead.foreach(_.resp.bits.data := 0.U)
  VecUnit.io.accRead.foreach(_.resp.bits.fromDMA := false.B)
  VecUnit.io.accWrite.foreach(_.req.ready := false.B)

  Im2col.io.sramRead.foreach(_.req.ready := false.B)
  Im2col.io.sramRead.foreach(_.resp.valid := false.B)
  Im2col.io.sramRead.foreach(_.resp.bits.data := 0.U)
  Im2col.io.sramRead.foreach(_.resp.bits.fromDMA := false.B)
  Im2col.io.sramWrite.foreach(_.req.ready := false.B)

// -----------------------------------------------------------------------------
// BBFP_Control
// -----------------------------------------------------------------------------

  // cmdReq输入分发已在上面处理

  val real_is_matmul_ws = WireInit(false.B)
  val reg_is_matmul_ws  = RegInit(false.B)
  
  when (io.cmdReq.ball2.valid) {
    reg_is_matmul_ws := io.cmdReq.ball2.bits.cmd.is_matmul_ws
  }
  real_is_matmul_ws := Mux(io.cmdReq.ball2.valid, io.cmdReq.ball2.bits.cmd.is_matmul_ws, reg_is_matmul_ws)

  BBFP_Control.io.is_matmul_ws := real_is_matmul_ws
  

  // 连接到Scratchpad的SRAM读写接口
  when (real_sel === bbfp_unit) {
    for (i <- 0 until b.sp_banks) {
      // sramRead(i).req 
      io.sramRead(i).req.valid              := BBFP_Control.io.sramRead(i).req.valid
      io.sramRead(i).req.bits               := BBFP_Control.io.sramRead(i).req.bits
      BBFP_Control.io.sramRead(i).req.ready := io.sramRead(i).req.ready

      // sramRead(i).resp 
      BBFP_Control.io.sramRead(i).resp.valid := io.sramRead(i).resp.valid
      BBFP_Control.io.sramRead(i).resp.bits  := io.sramRead(i).resp.bits
      io.sramRead(i).resp.ready              := BBFP_Control.io.sramRead(i).resp.ready

      // sramWrite(i) 
      io.sramWrite(i).req.valid     := BBFP_Control.io.sramWrite(i).req.valid
      io.sramWrite(i).req.bits.addr := BBFP_Control.io.sramWrite(i).req.bits.addr
      io.sramWrite(i).req.bits.data := BBFP_Control.io.sramWrite(i).req.bits.data
      io.sramWrite(i).req.bits.mask := BBFP_Control.io.sramWrite(i).req.bits.mask
    }

    // 连接到Accumulator的读写接口
    for (i <- 0 until b.acc_banks) {
      // accRead(i).req 
      io.accRead(i).req.valid              := BBFP_Control.io.accRead(i).req.valid
      io.accRead(i).req.bits               := BBFP_Control.io.accRead(i).req.bits
      BBFP_Control.io.accRead(i).req.ready := io.accRead(i).req.ready

      // accRead(i).resp 
      BBFP_Control.io.accRead(i).resp.valid := io.accRead(i).resp.valid 
      BBFP_Control.io.accRead(i).resp.bits  := io.accRead(i).resp.bits
      io.accRead(i).resp.ready              := BBFP_Control.io.accRead(i).resp.ready

      // accWrite(i) 
      io.accWrite(i).req.valid     := BBFP_Control.io.accWrite(i).req.valid
      io.accWrite(i).req.bits.addr := BBFP_Control.io.accWrite(i).req.bits.addr
      io.accWrite(i).req.bits.data := BBFP_Control.io.accWrite(i).req.bits.data
      io.accWrite(i).req.bits.mask := BBFP_Control.io.accWrite(i).req.bits.mask
    }

  }

// -----------------------------------------------------------------------------
// VecUnit
// -----------------------------------------------------------------------------

  // VecUnit cmdReq连接已在上面处理


    // 连接到Scratchpad的SRAM读写接口
  when (real_sel === vec_unit) {
    for (i <- 0 until b.sp_banks) {
      // sramRead(i).req
      io.sramRead(i).req.valid          := VecUnit.io.sramRead(i).req.valid
      io.sramRead(i).req.bits           := VecUnit.io.sramRead(i).req.bits
      VecUnit.io.sramRead(i).req.ready  := io.sramRead(i).req.ready

      // sramRead(i).resp
      VecUnit.io.sramRead(i).resp.valid := io.sramRead(i).resp.valid
      VecUnit.io.sramRead(i).resp.bits  := io.sramRead(i).resp.bits
      io.sramRead(i).resp.ready         := VecUnit.io.sramRead(i).resp.ready

      // sramWrite(i)
      io.sramWrite(i).req.valid     := VecUnit.io.sramWrite(i).req.valid
      io.sramWrite(i).req.bits.addr := VecUnit.io.sramWrite(i).req.bits.addr
      io.sramWrite(i).req.bits.data := VecUnit.io.sramWrite(i).req.bits.data
      io.sramWrite(i).req.bits.mask := VecUnit.io.sramWrite(i).req.bits.mask
    }

    // 连接到Accumulator的读写接口
    for (i <- 0 until b.acc_banks) {
      // accRead(i).req
      io.accRead(i).req.valid              := VecUnit.io.accRead(i).req.valid
      io.accRead(i).req.bits               := VecUnit.io.accRead(i).req.bits
      VecUnit.io.accRead(i).req.ready      := io.accRead(i).req.ready

      // accRead(i).resp
      VecUnit.io.accRead(i).resp.valid     := io.accRead(i).resp.valid
      VecUnit.io.accRead(i).resp.bits      := io.accRead(i).resp.bits
      io.accRead(i).resp.ready             := VecUnit.io.accRead(i).resp.ready

      // accWrite(i)
      io.accWrite(i).req.valid     := VecUnit.io.accWrite(i).req.valid
      io.accWrite(i).req.bits.addr := VecUnit.io.accWrite(i).req.bits.addr
      io.accWrite(i).req.bits.data := VecUnit.io.accWrite(i).req.bits.data
      io.accWrite(i).req.bits.mask := VecUnit.io.accWrite(i).req.bits.mask
    }
  }

// -----------------------------------------------------------------------------
// Im2col
// -----------------------------------------------------------------------------
  when (real_sel === im2col) {
    // 连接到Scratchpad的SRAM读写接口
      for (i <- 0 until b.sp_banks) {
        // sramRead(i).req
        io.sramRead(i).req.valid          := Im2col.io.sramRead(i).req.valid
        io.sramRead(i).req.bits           := Im2col.io.sramRead(i).req.bits
        Im2col.io.sramRead(i).req.ready  := io.sramRead(i).req.ready

        // sramRead(i).resp
        Im2col.io.sramRead(i).resp.valid := io.sramRead(i).resp.valid
        Im2col.io.sramRead(i).resp.bits  := io.sramRead(i).resp.bits
        io.sramRead(i).resp.ready         := Im2col.io.sramRead(i).resp.ready

        // sramWrite(i)
        io.sramWrite(i).req.valid     := Im2col.io.sramWrite(i).req.valid
        io.sramWrite(i).req.bits.addr := Im2col.io.sramWrite(i).req.bits.addr
        io.sramWrite(i).req.bits.data := Im2col.io.sramWrite(i).req.bits.data
        io.sramWrite(i).req.bits.mask := Im2col.io.sramWrite(i).req.bits.mask
      }
    }
  

// -----------------------------------------------------------------------------
// Output Selector
// -----------------------------------------------------------------------------
  // cmdResp输出分发
  io.cmdResp.ball1.valid   := VecUnit.io.cmdResp.valid
  io.cmdResp.ball1.bits    := VecUnit.io.cmdResp.bits
  VecUnit.io.cmdResp.ready := io.cmdResp.ball1.ready
  
  io.cmdResp.ball2.valid        := BBFP_Control.io.cmdResp.valid
  io.cmdResp.ball2.bits         := BBFP_Control.io.cmdResp.bits  
  BBFP_Control.io.cmdResp.ready := io.cmdResp.ball2.ready

  io.cmdResp.ball3.valid      := Im2col.io.cmdResp.valid
  io.cmdResp.ball3.bits       := Im2col.io.cmdResp.bits
  Im2col.io.cmdResp.ready    := io.cmdResp.ball3.ready
}
