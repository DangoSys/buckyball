package uvbb

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config._
import prototype.vector._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}

class VecBallTestWrapper(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)
  })

  val vecUnit = Module(new VecUnit)
  val cmdDpiC = Module(new CmdDpiC)

  // 连接VecUnit
  vecUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> vecUnit.io.cmdResp

  // 连接DPI-C接口
  cmdDpiC.io.cmdReq_valid := io.cmdReq.valid
  cmdDpiC.io.cmdReq_ready := io.cmdReq.ready
  cmdDpiC.io.cmdReq_bits_cmd_bid := io.cmdReq.bits.cmd.bid
  cmdDpiC.io.cmdReq_bits_cmd_iter := io.cmdReq.bits.cmd.iter
  cmdDpiC.io.cmdReq_bits_cmd_special := io.cmdReq.bits.cmd.special
  cmdDpiC.io.cmdReq_bits_rob_id := io.cmdReq.bits.rob_id

  cmdDpiC.io.cmdResp_valid := io.cmdResp.valid
  cmdDpiC.io.cmdResp_ready := io.cmdResp.ready
  cmdDpiC.io.cmdResp_bits_rob_id := io.cmdResp.bits.rob_id

  // SRAM连接
  val spad_w = b.veclane * b.inputType.getWidth
  for (i <- 0 until b.sp_banks) {
    val sramDpiC = Module(new SramDpiC(log2Up(b.spad_bank_entries), 8, spad_w / 8))

    vecUnit.io.sramRead(i).req.ready := true.B
    vecUnit.io.sramRead(i).resp.valid := RegNext(vecUnit.io.sramRead(i).req.valid)
    vecUnit.io.sramWrite(i).req.ready := true.B

    sramDpiC.io.addr := vecUnit.io.sramRead(i).req.bits.addr
    sramDpiC.io.ren := vecUnit.io.sramRead(i).req.valid
    sramDpiC.io.wdata := vecUnit.io.sramWrite(i).req.bits.data
    sramDpiC.io.wen := vecUnit.io.sramWrite(i).req.valid
    sramDpiC.io.mask := vecUnit.io.sramWrite(i).req.bits.mask

    vecUnit.io.sramRead(i).resp.bits.data := sramDpiC.io.rdata
  }

  // ACC连接
  for (i <- 0 until b.acc_banks) {
    val accDpiC = Module(new AccDpiC(log2Up(b.acc_bank_entries), 32, 4, b.acc_mask_len))

    vecUnit.io.accRead(i).req.ready := true.B
    vecUnit.io.accRead(i).resp.valid := RegNext(vecUnit.io.accRead(i).req.valid)
    vecUnit.io.accWrite(i).req.ready := true.B

    accDpiC.io.addr := vecUnit.io.accRead(i).req.bits.addr
    accDpiC.io.ren := vecUnit.io.accRead(i).req.valid
    accDpiC.io.wdata := vecUnit.io.accWrite(i).req.bits.data
    accDpiC.io.wen := vecUnit.io.accWrite(i).req.valid
    accDpiC.io.mask := vecUnit.io.accWrite(i).req.bits.mask

    vecUnit.io.accRead(i).resp.bits.data := accDpiC.io.rdata
  }
}
