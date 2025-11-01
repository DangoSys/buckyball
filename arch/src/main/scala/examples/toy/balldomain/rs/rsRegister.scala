package examples.toy.balldomain.rs

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallReservationStation, BallRsRegist}

/**
 * Ball RS module - references BBus mechanism, manages Ball device registration and connections
 */
class BallRSModule(implicit b: CustomBuckyBallConfig, p: Parameters) extends BallReservationStation(
  // Define Ball device information to register
  Seq(
    BallRsRegist(ballId = 0, ballName = "VecBall"),
    BallRsRegist(ballId = 1, ballName = "MatrixBall"),
    BallRsRegist(ballId = 2, ballName = "Im2colBall"),
    BallRsRegist(ballId = 3, ballName = "TransposeBall"),
    BallRsRegist(ballId = 4, ballName = "ReluBall"),
    BallRsRegist(ballId = 5, ballName = "EmptyBall")
  )
) {
  override lazy val desiredName = "BallRSModule"
}
