package examples.toy.balldomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.rs.{BallReservationStation, BallRsRegist}

/**
 * Ball RS module - references BBus mechanism, manages Ball device registration and connections
 */
@instantiable
class BallRSModule(parameter: BallDomainParam)
    extends BallReservationStation(
      parameter,
      // Define Ball device information to register
      Seq(
        BallRsRegist(ballId = 0, ballName = "VecBall"),
        BallRsRegist(ballId = 1, ballName = "MatrixBall"),
        BallRsRegist(ballId = 2, ballName = "Im2colBall"),
        BallRsRegist(ballId = 3, ballName = "TransposeBall"),
        BallRsRegist(ballId = 4, ballName = "ReluBall"),
        BallRsRegist(ballId = 5, ballName = "EmptyBall"),
        BallRsRegist(ballId = 6, ballName = "TransferBall")
      )
    ) {
  override lazy val desiredName = "BallRSModule"
}
