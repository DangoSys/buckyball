package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.blink.BallRegist
import framework.bbus.pmc.BallCyclePMC
import framework.bbus.cmdrouter.CmdRouter
import framework.bbus.memrouter.MemRouter
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
  io.sramRead <> memoryrouter.io.sramRead_o
  io.sramWrite <> memoryrouter.io.sramWrite_o
  io.accRead <> memoryrouter.io.accRead_o
  io.accWrite <> memoryrouter.io.accWrite_o
  memoryrouter.io.bbusConfig_i <> cmdRouter.io.bbusConfig_o
  
  // be replaced by ToVirtualLine and ToPhysicalLine modules
  //begin
  // for(i <- 0 until numBalls){
  //   memoryrouter.io.sramRead_i(i) <> balls(i).Blink.sramRead
  //   memoryrouter.io.sramWrite_i(i) <> balls(i).Blink.sramWrite
  //   memoryrouter.io.accRead_i(i) <> balls(i).Blink.accRead
  //   memoryrouter.io.accWrite_i(i) <> balls(i).Blink.accWrite
  // }
  //end

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


// -----------------------------------------------------------------------------
// ToPhysicalLine - per-ball conversion from virtual to physical line
// -----------------------------------------------------------------------------

  val toPhysicalLines = Seq.fill(numBalls){ Module(new ToPhysicalLine()(b, p)) }
  for (i <- 0 until numBalls) {
    toPhysicalLines(i).io.sramRead_i  <> toVirtualLines(i).io.sramRead_o
    toPhysicalLines(i).io.sramWrite_i <> toVirtualLines(i).io.sramWrite_o
    
    memoryrouter.io.sramRead_i(i)  <> toPhysicalLines(i).io.sramRead_o
    memoryrouter.io.sramWrite_i(i) <> toPhysicalLines(i).io.sramWrite_o
    memoryrouter.io.accRead_i(i)   <> toPhysicalLines(i).io.accRead_o
    memoryrouter.io.accWrite_i(i)  <> toPhysicalLines(i).io.accWrite_o
  }

  override lazy val desiredName = "BBus"
}
