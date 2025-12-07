package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.balldomain.blink.BallRegist
import framework.balldomain.bbus.pmc.BallCyclePMC
import framework.balldomain.bbus.cmdrouter.CmdRouter
import framework.balldomain.bbus.memrouter.MemRouter
import framework.switcher.{ToPhysicalLine, ToVirtualLine}


class BBusConfigIO(numBalls: Int)extends Bundle {
  val src_bid = UInt(log2Ceil(numBalls).W)
  val dst_bid = UInt(log2Ceil(numBalls).W)
  val set     = Bool()
}
/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices
 */
class BBus(ballGenerators: Seq[() => BallRegist with Module])
  (implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val numBalls = ballGenerators.length

  val io = IO(new Bundle {
    val cmdReq = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val cmdResp = Vec(numBalls, Decoupled(new BallRsComplete))

    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  // Instantiate all registered Balls
  val balls = ballGenerators.map(gen => Module(gen()))


// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------
  val cmdRouter = Module(new CmdRouter(numBalls))
  val idle_ball = Wire(Vec(numBalls, Bool()))
  for (i <- 0 until numBalls) {
    idle_ball(i) := balls(i).Blink.cmdReq.ready
  }

  cmdRouter.io.cmdReq_i <> io.cmdReq
  cmdRouter.io.ballIdle := idle_ball

  for (i <- 0 until numBalls) {
    balls(i).Blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).Blink.cmdReq.bits := cmdRouter.io.cmdReq_o.bits
  }

  cmdRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i =>
    balls(i).Blink.cmdReq.ready && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

  for (i <- 0 until numBalls) {
    cmdRouter.io.cmdResp_i(i) <> balls(i).Blink.cmdResp
  }

  io.cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// bus router
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// memory router
// -----------------------------------------------------------------------------
  val memoryrouter = Module(new MemRouter(numBalls)(b, p))
  memoryrouter.io.bbusConfig_i <> cmdRouter.io.bbusConfig_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  val pmc = Module(new BallCyclePMC(numBalls))

  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits := cmdRouter.io.cmdReq_i(i).bits
    // Remove delay caused by RoB blocking preventing commit
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits := cmdRouter.io.cmdResp_o(i).bits
  }

//-----------------------------------------------------------------------------
// ToVirtualLine - per-ball address to virtual line conversion
// -----------------------------------------------------------------------------

  val toVirtualLines = Seq.fill(numBalls){ Module(new ToVirtualLine()(b, p)) }
  for (i <- 0 until numBalls) {
    toVirtualLines(i).io.sramRead_i  <> balls(i).Blink.sramRead
    toVirtualLines(i).io.sramWrite_i <> balls(i).Blink.sramWrite
    toVirtualLines(i).io.accRead_i   <> balls(i).Blink.accRead
    toVirtualLines(i).io.accWrite_i  <> balls(i).Blink.accWrite
  }

  for(i <- 0 until numBalls){
    memoryrouter.io.sramRead_i(i) <> toVirtualLines(i).io.sramRead_o
    memoryrouter.io.sramWrite_i(i) <> toVirtualLines(i).io.sramWrite_o
  }

// -----------------------------------------------------------------------------
// ToPhysicalLine - per-ball conversion from virtual to physical line
// -----------------------------------------------------------------------------

  val toPhysicalLines = Module(new ToPhysicalLine()(b, p))
  
    toPhysicalLines.io.sramRead_i  <> memoryrouter.io.sramRead_o
    toPhysicalLines.io.sramWrite_i <> memoryrouter.io.sramWrite_o
    
    io.sramRead  <> toPhysicalLines.io.sramRead_o
    io.sramWrite <> toPhysicalLines.io.sramWrite_o
    io.accRead   <> toPhysicalLines.io.accRead_o
    io.accWrite  <> toPhysicalLines.io.accWrite_o

  override lazy val desiredName = "BBus"
}