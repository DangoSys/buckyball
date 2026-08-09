package framework.balldomain.isa

import framework.system.core.rocket.BuckyballCommand

/** Framework-reserved BallDomain instructions. */
object BallISA {
  val InitFunct = 0x05

  def init(ballId: Int): BuckyballCommand = {
    require(ballId >= 0 && ballId < 32, s"BALL_INIT target ID out of range: $ballId")
    BuckyballCommand(InitFunct, rs1 = 0, rs2 = BigInt(ballId))
  }

}
