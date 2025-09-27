package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp, SramWriteReq}

/**
 * 命令路由器 - 解析命令并输出目标Ball ID和路由后的命令
 * 优化版本：只对空闲的ball进行仲裁
 *
 * 输入：
 * - cmdReq：上游命令输入
 * - ballIdle：各ball的空闲状态信号
 *
 * 输出：
 * - ballId：目标Ball ID
 */
class memRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val sramRead_i = Vec(numBalls, Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite_i = Vec(numBalls, Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accRead_i = Vec(numBalls, Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite_i = Vec(numBalls, Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    val sramRead_o = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite_o = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accRead_o = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite_o = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  val memReq = WireInit(VecInit(Seq.fill(numBalls)(false.B))) // 显式初始化为false


  //默认赋值
  io.sramRead_o := DontCare
  io.sramWrite_o := DontCare
  io.accRead_o := DontCare
  io.accWrite_o := DontCare
  for (i <- 0 until numBalls) {
    io.sramRead_i(i).foreach(_.req.ready := false.B)
    io.sramRead_i(i).foreach(_.resp := DontCare)
    io.sramWrite_i(i).foreach(_.req.ready := false.B)
    io.accRead_i(i).foreach(_.req.ready := false.B)
    io.accRead_i(i).foreach(_.resp := DontCare)
    io.accWrite_i(i).foreach(_.req.ready := false.B)
  }

  //路由选择
  for (i <- 0 until numBalls) {

    memReq(i) := io.sramRead_i(i).map(_.req.valid).reduce(_||_) ||
                io.sramWrite_i(i).map(_.req.valid).reduce(_||_) ||
                io.accRead_i(i).map(_.req.valid).reduce(_||_)   ||
                io.accWrite_i(i).map(_.req.valid).reduce(_||_)  ||
                io.sramRead_i(i).map(_.resp.ready).reduce(_||_)

    when (memReq(i)) {
      io.sramRead_o <> io.sramRead_i(i)
      io.sramWrite_o <> io.sramWrite_i(i)
      io.accRead_o <> io.accRead_i(i)
      io.accWrite_o <> io.accWrite_i(i)
    }
  }

  /*
  val sramReadReq_arbiter = Module(new RRArbiter(Vec(b.sp_banks,new SramReadReq(b.spad_bank_entries)), numBalls))
  val sramWriteReq_arbiter = Module(new RRArbiter(Vec(b.sp_banks,new SramWriteReq(b.acc_bank_entries,b.spad_w, b.spad_mask_len)), numBalls))
  val accReadReq_arbiter = Module(new RRArbiter(Vec(b.sp_banks,new SramReadReq(b.acc_w)), numBalls))
  val accWriteReq_arbiter = Module(new RRArbiter(Vec(b.sp_banks,new SramWriteReq(b.acc_bank_entries,b.acc_w,b.acc_mask_len)), numBalls))
  for (i <- 0 until numBalls) {
    sramReadReq_arbiter.io.in(i).valid :=io.sramRead_i(i).map(_.req.valid).reduce(_||_) && io.ballIdle(i)
    sramWriteReq_arbiter.io.in(i).valid := io.sramWrite_i(i).map(_.req.valid).reduce(_||_) && io.ballIdle(i)
    for (j <- 0 until b.sp_banks) {
        sramReadReq_arbiter.io.in(i).bits(j) := io.sramRead_i(i)(j).req.bits
        io.sramRead_i(i)(j).req.ready := sramReadReq_arbiter.io.in(i).ready && io.ballIdle(i)

        sramWriteReq_arbiter.io.in(i).bits(j) := io.sramWrite_i(i)(j).req.bits
        io.sramWrite_i(i)(j).req.ready := sramWriteReq_arbiter.io.in(i).ready && io.ballIdle(i)
        io.sramWrite_o := sramWriteReq_arbiter.io.out
    }
  }

  for (i <- 0 until numBalls) {
    accReadReq_arbiter.io.in(i).valid := io.accRead_i(i).map(_.req.valid).reduce(_||_) && io.ballIdle(i)
    accWriteReq_arbiter.io.in(i).valid := io.accWrite_i(i).map(_.req.valid).reduce(_||_) && io.ballIdle(i)
    for (j <- 0 until b.acc_banks) {
        accReadReq_arbiter.io.in(i).bits(j) := io.accRead_i(i)(j).req.bits
        io.accRead_i(i)(j).req.ready := accReadReq_arbiter.io.in(i).ready && io.ballIdle(i)

        accWriteReq_arbiter.io.in(i).bits(j) := io.accWrite_i(i)(j).req.bits
        io.accWrite_i(i)(j).req.ready := accWriteReq_arbiter.io.in(i).ready && io.ballIdle(i)
    }
  }
*/

  override lazy val desiredName = "MemRouter"
}
