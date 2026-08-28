package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.HasBlink
import framework.balldomain.bbus.pmc.BallCyclePMC
import framework.balldomain.bbus.cmdrouter.CmdRouter
import framework.balldomain.isa.BallISA
import framework.balldomain.blink.{BankRead, BankWrite, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import java.lang.reflect.InvocationTargetException

/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices.
 *
 * Ball generators are produced reflectively from `b.ballDomain.ballIdMappings`:
 * each mapping carries a `ballClass` FQCN whose constructor `(GlobalConfig)` is
 * invoked. Framework does not maintain or interpret the list of balls; the
 * config layer is the single source of truth.
 */
@instantiable
class BBus(val b: GlobalConfig) extends Module {
  val numBalls       = b.ballDomain.ballNum
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val totalMmioRead  = b.ballDomain.ballIdMappings.map(_.mmioReadBW).sum
  val totalMmioWrite = b.ballDomain.ballIdMappings.map(_.mmioWriteBW).sum

  // Rs - bbus - balls
  @public
  val cmdReq            = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b)))))
  @public
  val cmdResp           = IO(Vec(numBalls, Decoupled(new BallRsComplete(b))))
  @public
  val ballChannelActive = IO(Output(Vec(numBalls, Bool())))
  @public
  val ballChannelReady  = IO(Input(Vec(numBalls, Bool())))
  // balls - bbus
  @public
  val bankRead          = IO(Vec(totalBallRead, Flipped(new BankRead(b))))
  @public
  val bankWrite         = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))
  @public
  val mmioRead          = IO(Vec(totalMmioRead, Flipped(new MmioRead(b))))
  @public
  val mmioWrite         = IO(Vec(totalMmioWrite, Flipped(new MmioWrite(b))))
  // balls - bbus - SubROB
  @public
  val subRobReq         = IO(Vec(numBalls, Decoupled(new SubRobRow(b))))

  require(b.ballDomain.ballIdMappings.length == numBalls, "ballNum must match ballIdMappings length")

  // Apply BALL_INIT on the cycle after the command handshake.  Driving reset
  // directly from cmd.fire would feed the Ball's reset-dependent ready signal
  // back into that same handshake.
  val ballBootReset = RegInit(VecInit(Seq.fill(numBalls)(false.B)))

  val balls = b.ballDomain.ballIdMappings.zipWithIndex.map { case (mapping, index) =>
    withReset(reset.asBool || ballBootReset(index)) {
      Module {
        try {
          val cls  = Class.forName(mapping.ballClass)
          val ctor = cls.getConstructor(classOf[GlobalConfig])
          ctor.newInstance(b).asInstanceOf[HasBlink with Module]
        } catch {
          case e: InvocationTargetException =>
            val cause = Option(e.getCause).getOrElse(e)
            throw new RuntimeException(
              s"Failed to instantiate ball ${mapping.ballName} (${mapping.ballClass}): ${cause.getMessage}",
              cause
            )
          case e: Throwable                 =>
            throw new RuntimeException(
              s"Failed to instantiate ball ${mapping.ballName} (${mapping.ballClass}): ${e.getMessage}",
              e
            )
        }
      }
    }
  }

  val cmdRouter: Instance[CmdRouter]    = Instantiate(new CmdRouter(b))
  val pmc:       Instance[BallCyclePMC] = Instantiate(new BallCyclePMC(b))

// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------

  val idle_ball = VecInit(balls.map(_.blink.cmdReq.ready))

  cmdRouter.io.cmdReq_i <> cmdReq
  cmdRouter.io.ballIdle := idle_ball

  val isBallInit    = cmdRouter.io.cmdReq_o.bits.cmd.funct7 === BallISA.InitFunct.U
  val initPending   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val initResp      = Reg(Vec(numBalls, new BallRsComplete(b)))
  val targetMatches = VecInit(b.ballDomain.ballIdMappings.map(m => cmdRouter.io.cmdReq_o.bits.cmd.bid === m.ballId.U))

  for (i <- 0 until numBalls) {
    ballChannelActive(i)        := balls(i).blink.status.running
    balls(i).blink.channelReady := ballChannelReady(i)

    val targetMatch = targetMatches(i)
    balls(i).blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && !isBallInit && targetMatch
    balls(i).blink.cmdReq.bits  := cmdRouter.io.cmdReq_o.bits

    cmdRouter.io.cmdResp_i(i).valid := Mux(initPending(i), true.B, balls(i).blink.cmdResp.valid)
    cmdRouter.io.cmdResp_i(i).bits  := Mux(initPending(i), initResp(i), balls(i).blink.cmdResp.bits)
    balls(i).blink.cmdResp.ready    := !initPending(i) && cmdRouter.io.cmdResp_i(i).ready

    ballBootReset(i) := cmdRouter.io.cmdReq_o.fire && isBallInit && targetMatch
    when(cmdRouter.io.cmdReq_o.fire && isBallInit && targetMatch) {
      initPending(i)         := true.B
      initResp(i).rob_id     := cmdRouter.io.cmdReq_o.bits.rob_id
      initResp(i).is_sub     := cmdRouter.io.cmdReq_o.bits.is_sub
      initResp(i).sub_rob_id := cmdRouter.io.cmdReq_o.bits.sub_rob_id
    }
    when(initPending(i) && cmdRouter.io.cmdResp_i(i).fire) {
      initPending(i) := false.B
    }
  }

  val targetReady = VecInit((0 until numBalls).map(i => balls(i).blink.cmdReq.ready && targetMatches(i))).asUInt.orR
  cmdRouter.io.cmdReq_o.ready := Mux(isBallInit, targetMatches.asUInt.orR, targetReady)

  cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

// Connect balls' bankRead and bankWrite to memrouter
  var readChannelIdx  = 0
  var writeChannelIdx = 0

  for (ball <- balls) {
    val ballConfig = b.ballDomain.ballIdMappings.find(_.ballName == ball.getClass.getSimpleName)
    val inBW       = ballConfig.map(_.inBW).getOrElse(0)
    val outBW      = ballConfig.map(_.outBW).getOrElse(0)

    for (i <- 0 until inBW) {
      bankRead(readChannelIdx) <> ball.blink.bankRead(i)
      readChannelIdx = readChannelIdx + 1
    }

    for (i <- 0 until outBW) {
      bankWrite(writeChannelIdx) <> ball.blink.bankWrite(i)
      writeChannelIdx = writeChannelIdx + 1
    }
  }

  var mmioWriteChannelIdx = 0
  for (ball <- balls) {
    val mmioWriteBW = b.ballDomain.ballIdMappings
      .find(_.ballName == ball.getClass.getSimpleName)
      .map(_.mmioWriteBW)
      .getOrElse(0)
    for (i <- 0 until mmioWriteBW) {
      mmioWrite(mmioWriteChannelIdx) <> ball.blink.mmioWrite(i)
      mmioWriteChannelIdx = mmioWriteChannelIdx + 1
    }
  }

  // Connect balls' subRobReq
  for (i <- 0 until numBalls) {
    subRobReq(i) <> balls(i).blink.subRobReq
  }

  // Connect configurable MMIO metadata channels.
  var mmioReadChannelIdx = 0
  for (ball <- balls) {
    val mmioReadBW = b.ballDomain.ballIdMappings
      .find(_.ballName == ball.getClass.getSimpleName)
      .map(_.mmioReadBW)
      .getOrElse(0)
    for (i <- 0 until mmioReadBW) {
      mmioRead(mmioReadChannelIdx) <> ball.blink.mmioRead(i)
      mmioReadChannelIdx = mmioReadChannelIdx + 1
    }
  }

}
