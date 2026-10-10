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
  val cmdReq                   = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b)))))
  @public
  val cmdResp                  = IO(Vec(numBalls, Decoupled(new BallRsComplete(b))))
  @public
  val ballChannelActive        = IO(Output(Vec(numBalls, Bool())))
  @public
  val ballChannelReady         = IO(Input(Vec(numBalls, Bool())))
  // balls - bbus
  @public
  val bankRead                 = IO(Vec(totalBallRead, Flipped(new BankRead(b))))
  @public
  val bankWrite                = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))
  @public
  val mmioRead                 = IO(Vec(totalMmioRead, Flipped(new MmioRead(b))))
  @public
  val mmioWrite                = IO(Vec(totalMmioWrite, Flipped(new MmioWrite(b))))
  // balls - bbus - SubROB
  @public
  val subRobReq                = IO(Vec(numBalls, Decoupled(new SubRobRow(b))))
  @public val kernelBanks      = if (b.rvv.enable) Some(IO(Input(new framework.rvv.BankLayout(b)))) else None
  @public val internalCommand  = if (b.rvv.enable) Some(IO(Flipped(Decoupled(new BallRsIssue(b))))) else None
  @public val internalComplete = if (b.rvv.enable) Some(IO(Decoupled(UInt(64.W)))) else None

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

  val command       = Wire(Decoupled(new BallRsIssue(b)))
  val internal      = WireDefault(false.B)
  val internalOwner = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  if (b.rvv.enable) {
    val arbiter = Module(new RRArbiter(new BallRsIssue(b), 2))
    arbiter.io.in(0) <> cmdRouter.io.cmdReq_o
    arbiter.io.in(1) <> internalCommand.get
    command <> arbiter.io.out
    internal                   := arbiter.io.chosen === 1.U
    internalComplete.get.valid := VecInit(
      (0 until numBalls).map(i => internalOwner(i) && balls(i).blink.cmdResp.valid)
    ).asUInt.orR
    internalComplete.get.bits  := 0.U
    assert(PopCount(internalOwner) <= 1.U, "RVV issues only one outstanding Ball operation")
  } else {
    command <> cmdRouter.io.cmdReq_o
  }

  val isBallInit    = command.bits.cmd.funct7 === BallISA.InitFunct.U
  val initPending   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val initResp      = Reg(Vec(numBalls, new BallRsComplete(b)))
  val targetMatches = VecInit(b.ballDomain.ballIdMappings.map(m => command.bits.cmd.bid === m.ballId.U))

  for (i <- 0 until numBalls) {
    val mapping = b.ballDomain.ballIdMappings(i)
    ballChannelActive(i)        := (if (mapping.inBW == 0 && mapping.outBW == 0) false.B else balls(i).blink.status.running)
    balls(i).blink.channelReady := ballChannelReady(i)

    val targetMatch = targetMatches(i)
    balls(i).blink.cmdReq.valid := command.valid && !isBallInit && !initPending(i) && targetMatch
    balls(i).blink.cmdReq.bits  := command.bits

    cmdRouter.io.cmdResp_i(i).valid                    := !internalOwner(i) && Mux(initPending(i), true.B, balls(i).blink.cmdResp.valid)
    cmdRouter.io.cmdResp_i(i).bits                     := Mux(initPending(i), initResp(i), balls(i).blink.cmdResp.bits)
    balls(i).blink.cmdResp.ready                       := !initPending(i) && Mux(
      internalOwner(i),
      (if (b.rvv.enable) internalComplete.get.ready else false.B),
      cmdRouter.io.cmdResp_i(i).ready
    )
    when(command.fire && targetMatch)(internalOwner(i) := internal)
    when(balls(i).blink.cmdResp.fire || initPending(i) && cmdRouter.io.cmdResp_i(i).fire) {
      internalOwner(i) := false.B
    }
    if (b.rvv.enable) {
      when(internalOwner(i) && initPending(i)) {
        internalComplete.get.valid := true.B
        when(internalComplete.get.fire) { initPending(i) := false.B; internalOwner(i) := false.B }
      }
    }

    ballBootReset(i) := command.fire && isBallInit && targetMatch
    when(command.fire && isBallInit && targetMatch) {
      initPending(i)         := true.B
      initResp(i).rob_id     := command.bits.rob_id
      initResp(i).is_sub     := command.bits.is_sub
      initResp(i).sub_rob_id := command.bits.sub_rob_id
    }
    when(initPending(i) && cmdRouter.io.cmdResp_i(i).fire) {
      initPending(i) := false.B
    }
  }

  val targetReady =
    VecInit((0 until numBalls).map(i => balls(i).blink.cmdReq.ready && !initPending(i) && targetMatches(i))).asUInt.orR
  command.ready := targetReady

  cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire || command.fire && internal && targetMatches(i)
    pmc.io.cmdReq_i(i).bits   := Mux(internal, command.bits, cmdRouter.io.cmdReq_i(i).bits)
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid ||
      internalOwner(i) && (balls(i).blink.cmdResp.fire || initPending(i) && (if (b.rvv.enable) internalComplete.get.fire
                                                                             else false.B))
    pmc.io.cmdResp_o(i).bits  := Mux(
      internalOwner(i),
      Mux(initPending(i), initResp(i), balls(i).blink.cmdResp.bits),
      cmdRouter.io.cmdResp_o(i).bits
    )
  }

// Connect balls' bankRead and bankWrite to memrouter
  var readChannelIdx  = 0
  var writeChannelIdx = 0

  for (((mapping, ball), index) <- b.ballDomain.ballIdMappings.zip(balls).zipWithIndex) {
    val inBW  = mapping.inBW
    val outBW = mapping.outBW

    for (i <- 0 until inBW) {
      bankRead(readChannelIdx) <> ball.blink.bankRead(i)
      if (b.rvv.enable) {
        val local  = ball.blink.bankRead(i).bank_id
        val layout = kernelBanks.get
        when(internalOwner(index)) {
          bankRead(readChannelIdx).bank_id  := Mux(local < layout.readGroups, layout.readBank, layout.writeBank)
          bankRead(readChannelIdx).group_id := Mux(local < layout.readGroups, local, local - layout.readGroups)
          when(ball.blink.bankRead(i).io.req.valid) {
            assert(local < (layout.readGroups +& layout.writeGroups), "RVV Ball read bank outside launch groups")
          }
        }
      }
      readChannelIdx = readChannelIdx + 1
    }

    for (i <- 0 until outBW) {
      bankWrite(writeChannelIdx) <> ball.blink.bankWrite(i)
      if (b.rvv.enable) {
        val local  = ball.blink.bankWrite(i).bank_id
        val layout = kernelBanks.get
        when(internalOwner(index)) {
          bankWrite(writeChannelIdx).bank_id  := layout.writeBank
          bankWrite(writeChannelIdx).group_id := local - layout.readGroups
          when(ball.blink.bankWrite(i).io.req.valid) {
            assert(
              local >= layout.readGroups && local < (layout.readGroups +& layout.writeGroups),
              "RVV Ball write outside write groups"
            )
          }
        }
      }
      writeChannelIdx = writeChannelIdx + 1
    }
  }

  var mmioWriteChannelIdx = 0
  for ((mapping, ball) <- b.ballDomain.ballIdMappings.zip(balls)) {
    val mmioWriteBW = mapping.mmioWriteBW
    for (i <- 0 until mmioWriteBW) {
      mmioWrite(mmioWriteChannelIdx) <> ball.blink.mmioWrite(i)
      mmioWriteChannelIdx = mmioWriteChannelIdx + 1
    }
  }

  // Internal RVV operations complete directly; they cannot expand into SubROB.
  for (i <- 0 until numBalls) {
    subRobReq(i).valid             := balls(i).blink.subRobReq.valid && !internalOwner(i)
    subRobReq(i).bits              := balls(i).blink.subRobReq.bits
    balls(i).blink.subRobReq.ready := subRobReq(i).ready && !internalOwner(i)
    assert(!(internalOwner(i) && balls(i).blink.subRobReq.valid), "RVV Ball operation cannot issue SubROB work")
  }

  // Connect configurable MMIO metadata channels.
  var mmioReadChannelIdx = 0
  for ((mapping, ball) <- b.ballDomain.ballIdMappings.zip(balls)) {
    val mmioReadBW = mapping.mmioReadBW
    for (i <- 0 until mmioReadBW) {
      mmioRead(mmioReadChannelIdx) <> ball.blink.mmioRead(i)
      mmioReadChannelIdx = mmioReadChannelIdx + 1
    }
  }

}
